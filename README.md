# Tensorflow Trees
Library refactored out of my master thesis work. It efficiently implements the encoding and decoding of tree structured data, employing Tensorflow Eager Mode to deal with the dynamical nature of the computations.

## Example
In `examples\simple_expression` is provided a well documented example an autoencoder for tree structured arithmetic expressions.

In order to run it fullfill all the dependencies in `requirements.txt`, for instance with conda:  
`conda create --name tf_tree --file requirements.txt`

Then you can run the example with all the default settings as:  
`PYTHONPATH=. python examples/simple_expression/exp_autoencoder.py`  
 Use `--helpfull` for all the available flags.
 
 ## Concepts
 In order to be able to handle tree structured data some concepts need to be introduced.
 
 ### Trees Definition
 First of all we need a way to characterize the trees we want to deal with. So we can instantiate the proper sub-networks and generate valid trees.  
 See `examples/simple_expression/exp_definition` for an example.
 ### Batch
 ### Encoder
 ### Decoder

 ### More
 For a in depth discussion you can refer to my [master thesis](https://github.com/m-colombo/conditional-variational-tree-autoencoder/blob/master/thesis.pdf) about conditional variational autoencoder on tree structured data.
 
 ## Benchmark
 
 ## Dev status