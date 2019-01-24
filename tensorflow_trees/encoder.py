import tensorflow as tf
import typing as T

from tensorflow_trees.definition import TreeDefinition, Tree, NodeDefinition
from tensorflow_trees.batch import BatchOfTreesForEncoding
from tensorflow_trees.encoder_cells_builder import EncoderCellsBuilder


class Encoder(tf.keras.Model):
    def __init__(self, *,
                 tree_def: TreeDefinition = None, embedding_size: int = None, cellsbuilder: EncoderCellsBuilder = None, cut_arity: int = None, max_arity = None, name='',
                 variable_arity_strategy="FLAT"
                 ):
        """

        :param tree_def:
        :param embedding_size:
        """
        super(Encoder, self).__init__()

        self.tree_def = tree_def
        self.node_map = {n.id: n for n in tree_def.node_types}

        self.use_flat_strategy = variable_arity_strategy == "FLAT"
        self.max_arity = max_arity
        self.cut_arity = cut_arity

        self.embedding_size = embedding_size

        # if not attr, they don't get registered as variable by the keras model (dunno why)
        for t in tree_def.node_types:
            if self.use_flat_strategy and type(t.arity) == t.VariableArity:
                c, e = cellsbuilder.build_cell(t, self, name=name+"C_" + t.id)
                setattr(self, 'C_'+t.id, c)
                setattr(self, 'C_extra_' + t.id, e)

            elif not (type(t.arity) == t.FixedArity and t.arity.value == 0):
                setattr(self, 'C_'+t.id, cellsbuilder.build_cell(t, self, name=name+"C_" + t.id))

            if t.value_type is not None:
                setattr(self, 'M_' + t.id, cellsbuilder.build_merger(t, embedding_size, name=name+"M_" + t.id))

        for l in tree_def.leaves_types:
            setattr(self, 'E_'+l.id, cellsbuilder.build_embedder(l, embedding_size, name=name+"E_" + l.id))


    def _c_fixed_op(self, inp, node_id, ops, network):

        res = network.optimized_call(inp)

        if self.node_map[node_id].value_type is None:

            ops[0].meta['emb_batch'].scatter_update('embs', [op.meta['node_numb'] for op in ops], res)
            for op in ops:
                op.meta['computed'] = True

        # compute merging with value straight on - input already in place
        else:
            values = self.node_map[node_id].value_type.abstract_to_representation_batch([x.value.abstract_value for x in ops])
            self._m_op(res, values, getattr(self, 'M_'+node_id), ops)

        # recursive fused call is ignored - not useful in our use cases

    def _m_op(self, embs, values, network, ops):

            inp = tf.concat([embs, values], axis=-1)
            res = network.optimized_call(inp)

            ops[0].meta['emb_batch'].scatter_update('embs', [op.meta['node_numb'] for op in ops], res)

            for op in ops:
                op.meta['computed'] = True

    def __call__(self, batch: T.Union[BatchOfTreesForEncoding, T.List[Tree]]) -> BatchOfTreesForEncoding:

        if not type(batch) == BatchOfTreesForEncoding:
            batch = BatchOfTreesForEncoding(batch, self.embedding_size)

        all_ops = {}    # (op_type, node_id) -> [inputs]

        # 1) visit the trees, store leaves computation and create back-links to traverse the tree bottom-up
        def init(node, **kwargs):

            node.meta['computed'] = False

            if 'added' in node.meta.keys():
                del node.meta['added']

            # leaves are always computable with no dependencies
            if len(node.children) == 0:

                if ('E', node.node_type_id) not in all_ops:
                    all_ops[('E', node.node_type_id)] = []

                # store operation
                all_ops[('E', node.node_type_id)].append(node)

            else:
                # children recursion and back-links
                for c in node.children:
                    c.meta['parent'] = node

        batch.map_to_all_nodes(init, batch.trees)

        while len(all_ops) > 0:
            # 2) compute the aggregated most-required computation

            op_t, node_id = max(all_ops.keys(), key=lambda k: len(all_ops[k]))
            ops = all_ops.pop((op_t, node_id))
            if len(ops) == 0:
                break

            node_t = self.node_map[node_id]
            network = getattr(self, op_t + '_' + node_t.id)

            if op_t == 'E':

                inp = node_t.value_type.abstract_to_representation_batch([x.value.abstract_value for x in ops])
                res = network.optimized_call(inp)

                #  superflous when node is fused
                if node_id not in self.tree_def.fusable_nodes_id_child_parent.keys():
                    batch.scatter_update('embs',
                                      [op.meta['node_numb'] for op in ops],
                                      res)

                    for op in ops:
                        op.meta['computed'] = True

                else:
                    rec_ops = [o.meta['parent'] for o in ops]
                    network = getattr(self, 'C_'+ops[0].meta['parent'].node_type_id)
                    self._c_fixed_op(res, ops[0].meta['parent'].node_type_id, rec_ops, network)

            elif op_t == 'M':
                    values = node_t.value_type.abstract_to_representation_batch([x.value.abstract_value for x in ops])
                    embs = tf.gather(batch['embs'], [x.meta['node_numb'] for x in ops])
                    self._m_op(embs, values, network, ops)

            elif op_t == 'C':
                if type(node_t.arity) == NodeDefinition.FixedArity:
                    inp = tf.gather(batch['embs'], [c.meta['node_numb'] for op in ops for c in op.children])
                    inp_1 = tf.reshape(inp, [len(ops), -1])
                    self._c_fixed_op(inp_1, node_id, ops, network)

                elif type(node_t.arity) == NodeDefinition.VariableArity and not self.use_flat_strategy:
                    # TODO rnn (e.g. lstm) ?

                    idx = tf.cast(tf.reshape([[o.meta['node_numb'], o.children[o.meta.get('next_child', 0)].meta['node_numb']]
                                              for o in ops], [-1]), tf.int64)

                    inp = tf.gather(batch['embs'], idx)

                    inp = tf.reshape(inp, [len(ops), -1])
                    res = network.optimized_call(inp)

                    k = ('C', node_id)
                    if k not in all_ops.keys():
                        all_ops[k] = []

                    # add results
                    batch.scatter_update('embs',
                                      [op.meta['node_numb'] for op in ops],
                                      res)

                    for o in ops:

                        if o.meta.get('next_child', 0) == len(o.children) - 1:  # merged the last children
                            if self.node_map[node_id].value_type is not None:
                                all_ops[k].append(o)
                            else:
                                o.meta['computed'] = True
                        else:   # keep computing
                            o.meta['next_child'] = o.meta.get('next_child', 0) + 1
                            o.meta.pop('added')

                            # keep computing if other child is ready
                            if o.children[o.meta['next_child']].meta['computed']:
                                if ('C', o.node_type_id) not in all_ops:
                                    all_ops[('C', o.node_type_id)] = []

                                all_ops[('C', o.node_type_id)].append(o)
                                # avoid computing multiple times the parent when multiple children have been computed ad once
                                o.meta['added'] = None
                elif type(node_t.arity) == NodeDefinition.VariableArity and self.use_flat_strategy:
                    # TODO warning if max_arity < actual_arity (the code won't work anyway - crash)

                    max_children_arity = max([len(o.children) for o in ops])
                    first_arity = min(max_children_arity, self.cut_arity)
                    EXTRA_CHILD = max_children_arity > self.cut_arity

                    # extra children
                    if EXTRA_CHILD:
                        extra_network = getattr(self, op_t + '_extra_' + node_t.id)
                        # TODO we can use less zero padding
                        extra_inp = tf.gather(batch['embs'],tf.reshape(tf.convert_to_tensor([[c.meta['node_numb'] for c in o.children[self.cut_arity:]] + ([0] * (max_children_arity - len(o.children))) for o in ops if len(o.children) > self.cut_arity]), [-1]))
                        weights = extra_network(extra_inp)
                        weights = tf.reshape(tf.nn.softmax(tf.reshape(weights, [-1, max_children_arity - self.cut_arity])), [-1, max_children_arity - self.cut_arity, 1])
                        extra_inp = tf.reshape(extra_inp, [-1, max_children_arity - self.cut_arity, self.embedding_size])
                        weighted = tf.reduce_sum(weights * extra_inp, axis=1)
                        indeces = [[i] for o, i in zip(ops, range(len(ops))) if len(o.children) > self.cut_arity]
                        extra_children = tf.scatter_nd(indeces, weighted, [len(ops), self.embedding_size])
                    else:
                        extra_children = tf.zeros([len(ops), self.embedding_size])

                    # first children
                    inp = tf.gather(batch['embs'], tf.reshape(tf.convert_to_tensor([[c.meta['node_numb'] for c in o.children[:self.cut_arity]] + ([0] * (first_arity - len(o.children))) for o in ops]), [-1]))
                    inp = tf.reshape(inp, [len(ops), first_arity * self.embedding_size])
                    inp = tf.concat([inp, extra_children], axis=1)
                    res = network.optimized_call(inp)

                    if node_t.value_type is None:
                        batch.scatter_update('embs',
                                          [op.meta['node_numb'] for op in ops],
                                          res)
                        for o in ops:
                            o.meta['computed'] = True

                    else:
                            m_net = getattr(self, 'M_' + node_id)
                            inp = node_t.value_type.abstract_to_representation_batch([x.value.abstract_value for x in ops])
                            self._m_op(res, inp, m_net, ops)
            else:
                raise NotImplementedError()

            # 3) find new ready to go operations
            # res should be: [number_of_ops x output_size], with order preserved as in all_ops
            for op in ops:

                # use back-link to find ready-to-go operations
                if "parent" in op.meta.keys():  # otherwise is the root
                    parent = op.meta['parent']

                    # keep bubbling up if we already computed the parent (might be some fused op)
                    while parent.meta['computed'] and 'parent' in parent.meta.keys():
                        parent = parent.meta['parent']
                    if parent.meta['computed']:
                        continue    # we reached the root and it's computed, we're done for this tree

                    if ('added' not in parent.meta.keys() and
                        # For fixed arity node we need all children to be computed
                        (((type(self.node_map[parent.node_type_id].arity) == NodeDefinition.FixedArity or self.use_flat_strategy)
                            and all(map(lambda s: s.meta['computed'], parent.children)))
                        or
                        # For variable arity node we just need the next node
                        (type(self.node_map[parent.node_type_id].arity) == NodeDefinition.VariableArity
                            and not self.use_flat_strategy
                            and parent.children[parent.meta.get('next_child', 0)].meta['computed']))):

                        if ('C', parent.node_type_id) not in all_ops:
                            all_ops[('C', parent.node_type_id)] = []

                        all_ops[('C', parent.node_type_id)].append(parent)

                        # avoid computing multiple times the parent when multiple children have been computed ad once
                        parent.meta['added'] = None

        return batch
