# Tensorflow Trees
A library to deal with tree-structured data in TensorFlow. It provides an efficient implementation of an Encoder and a Decoder mappings trees to and from a flat space.

## Example
In `examples\simple_expression` is provided a well documented example of an autoencoder for tree structured arithmetic expressions.

In order to run it fullfill all the dependencies in `requirements.txt`, for instance with conda:  
`conda create --name tf_tree --file requirements.txt`

Then you can run the example with all the default settings as:  
`PYTHONPATH=. python examples/simple_expression/exp_autoencoder.py`  
 Use `--helpfull` for all the available flags.
 
 ## Concepts
 In order to be able to handle tree structured data some concepts need to be introduced.
 
 ### Trees Definition
 First of all we need a way to characterize the trees we want to deal with. So we can instantiate the proper sub-networks and generate valid trees.  
 See `examples/simple_expression/exp_definition` for a full example.
 
 A tree is characterized by a `TreeDefinition` object, basically listing the kind of nodes can appear in the trees:
 
 ```python
TreeDefinition(node_types=[NODE_DEF1, NODE_DEF2, NODE_DEF3, ...])
```

Every node is characterized by a `NodeDefinition` object:
 - associating an unique string id for the node type
 - defining whether such nodes can appear as root nodes
 - characterizing their arity (the number of children they have)
 - characterizing the associated value they might have
 
 For instance:
```python
NodeDefinition("node_type_id", may_root=True, arity=NodeDefinition.FixedArity(0), value_type=VALUE_TYPE_DEF)
```

Every value appearing associated to a node must be characterized by extending the class `NodeDefinition.Value` implementing:
- `representation_to_abstract_batch` and `abstract_to_representation_batch` methods to convert values from abstract to representation and viceversa. i.e. human readable and neural network readable.
- `representation_shape` indicating the size of the representation-
- `class_value` denoting whether the value is one hot encoding or a dense embedding 

 ### Batch
 Due to structured nature of the computation we need some support data structure, `BatchOfTreesForEncoding` and `BatchOfTreesForDecoding` defined in `tensorflow_trees/batch.py`.
 They are used to efficiently and incrementally store trees and intermediate values during the computations.
 
 ### Encoder
 TODO
 ### Decoder
 TODO
 
 ### More
 For a in depth discussion you can refer to my [master thesis](https://github.com/m-colombo/conditional-variational-tree-autoencoder/blob/master/thesis.pdf) about conditional variational autoencoder on tree structured data. 
 Although some details differ. 
 
 ## Benchmark
 TODO
 
 ## Dev status
 TODO