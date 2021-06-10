# orb-tensorflowjs
*orb-tensorflowjs* exposes three sets of APIs as follows:
* Input APIs transform the input arrangement.
* Layer APIs define the interaction of input and layers.
* Tensor APIs generate commonly used tensors.

These APIs work together to simplfy the model design. We will go through a few examples later on. Input and Layer APIs expose *apply()* method which performs the actual transformation.

# Installation
Browser Installation. The module is exported as *orbtfjs* global variable.

```html
<script src="https://cdn.jsdelivr.net/npm/orb-tensorflowjs@1.0.0/dist/index.js"></script>
```

Node Installation
```js
npm install orb-tensorflowjs
```

# Model Design
### A Sequential model with an input and a softmax layer.
*serial* pseudo layer constructs a sequential flow of its arg layers. The input is applied with a call to *apply()* method. The output of an arg layer is fed as input to the next arg layer. The output of the last arg layer is returned.

```js
const {serial } = require('orb-tensorflowjs')

const input = tf.input({shape: [4, 4]})
const output = serial(tf.layers.dense({units: 10, activation: 'softmax'}))
.apply(input)

const m = tf.model({inputs: input, outputs: output})
m.summary()
```
<pre>
_________________________________________________________________
 Layer (type)                 Output shape              Param #   
=================================================================
 input1 (InputLayer)          [null,4,4]                0         
_________________________________________________________________
 dense_Dense1 (Dense)         [null,4,10]               50        
=================================================================
Total params: 50
Trainable params: 50
Non-trainable params: 0
_________________________________________________________________
</pre>

### Feed an input to multiple layers in parallel
*parallel* pseudo layer feeds its input to each arg layer in parallel. The input is applied with a call to *apply()* method. It outputs an array containing the output from each arg layer.

```js
const {serial, parallel } = require('orb-tensorflowjs')

const input = tf.input({shape: [4, 4]})
const output = serial(
  parallel(
    tf.layers.dense({units: 10}),
    tf.layers.dense({units: 10})
  ),
  tf.layers.concatenate(),
  tf.layers.dense({units: 10, activation: 'softmax'})
).apply(input)
const m = tf.model({inputs: input, outputs: output})
m.summary()
```
<pre>
__________________________________________________________________________________________________
 Layer (type)                    Output shape         Param #     Receives inputs                  
==================================================================================================
 input1 (InputLayer)             [null,4,4]           0                                            
__________________________________________________________________________________________________
 dense_Dense1 (Dense)            [null,4,10]          50          input1[0][0]                     
__________________________________________________________________________________________________
 dense_Dense2 (Dense)            [null,4,10]          50          input1[0][0]                     
__________________________________________________________________________________________________
 concatenate_Concatenate1 (Conca [null,4,20]          0           dense_Dense1[0][0]               
                                                                  dense_Dense2[0][0]               
__________________________________________________________________________________________________
 dense_Dense3 (Dense)            [null,4,10]          210         concatenate_Concatenate1[0][0]   
==================================================================================================
Total params: 310
Trainable params: 310
Non-trainable params: 0
__________________________________________________________________________________________________
</pre>

### Split Example
*split* pseudo layer splits the input. It supports configurations to tweak the splits.

```js
const {serial, split } = require('orb-tensorflowjs')

const input = tf.input({shape: [4, 4]})
const output = serial(
  split(),
  tf.layers.concatenate(),
  tf.layers.dense({units: 10, activation: 'softmax'})
).apply(input)
const m = tf.model({inputs: input, outputs: output})
m.summary()
```
<pre>
__________________________________________________________________________________________________
 Layer (type)                    Output shape         Param #     Receives inputs                  
==================================================================================================
 input1 (InputLayer)             [null,4,4]           0                                            
__________________________________________________________________________________________________
 split_Split1 (Split)            [[null,1,4],[null,1, 0           input1[0][0]                     
__________________________________________________________________________________________________
 concatenate_Concatenate1 (Conca [null,1,16]          0           split_Split1[0][0]               
                                                                  split_Split1[0][1]               
                                                                  split_Split1[0][2]               
                                                                  split_Split1[0][3]               
__________________________________________________________________________________________________
 dense_Dense1 (Dense)            [null,1,10]          170         concatenate_Concatenate1[0][0]   
==================================================================================================
Total params: 170
Trainable params: 170
Non-trainable params: 0
__________________________________________________________________________________________________
</pre>

### Map elements in an input array individually to a layer
*mapTo* feeds the individual elements of an input array to the same layer.

```js
const {serial, split, mapTo } = require('orb-tensorflowjs')

const input = tf.input({shape: [4, 4]})
const ml = tf.layers.dense({units: 5})
const output = serial(
  split(),
  mapTo(ml),
  tf.layers.concatenate(),
  tf.layers.dense({units: 10, activation: 'softmax'})
).apply(input)
const m = tf.model({inputs: input, outputs: output})
m.summary()
```
<pre>
__________________________________________________________________________________________________
 Layer (type)                    Output shape         Param #     Receives inputs                  
==================================================================================================
 input1 (InputLayer)             [null,4,4]           0                                            
__________________________________________________________________________________________________
 split_Split1 (Split)            [[null,1,4],[null,1, 0           input1[0][0]                     
__________________________________________________________________________________________________
 dense_Dense1 (Dense)            [null,1,5]           25          split_Split1[0][0]               
                                                                  split_Split1[0][1]               
                                                                  split_Split1[0][2]               
                                                                  split_Split1[0][3]               
__________________________________________________________________________________________________
 concatenate_Concatenate1 (Conca [null,1,20]          0           dense_Dense1[0][0]               
                                                                  dense_Dense1[1][0]               
                                                                  dense_Dense1[2][0]               
                                                                  dense_Dense1[3][0]               
__________________________________________________________________________________________________
 dense_Dense2 (Dense)            [null,1,10]          210         concatenate_Concatenate1[0][0]   
==================================================================================================
Total params: 235
Trainable params: 235
Non-trainable params: 0
__________________________________________________________________________________________________
</pre>

### One-to-One mapping of an input element to the layer arg
*split()* pseudo layer splits the input into 2 components. *map()* pseudo layer maps each split to a dense layer.
```js
const {serial, split, map } = require('orb-tensorflowjs')

const input = tf.input({shape: [4, 4]})
const output = serial(
  split({factor: 2}),
  map(
    tf.layers.dense({units: 10}),
    tf.layers.dense({units: 10}),
  ),
  tf.layers.concatenate(),
  tf.layers.dense({units: 10, activation: 'softmax'})
).apply(input)
const m = tf.model({inputs: input, outputs: output})
m.summary()
```
<pre>
__________________________________________________________________________________________________
 Layer (type)                    Output shape         Param #     Receives inputs                  
==================================================================================================
 input1 (InputLayer)             [null,4,4]           0                                            
__________________________________________________________________________________________________
 split_Split1 (Split)            [[null,2,4],[null,2, 0           input1[0][0]                     
__________________________________________________________________________________________________
 dense_Dense1 (Dense)            [null,2,10]          50          split_Split1[0][0]               
__________________________________________________________________________________________________
 dense_Dense2 (Dense)            [null,2,10]          50          split_Split1[0][1]               
__________________________________________________________________________________________________
 concatenate_Concatenate1 (Conca [null,2,20]          0           dense_Dense1[0][0]               
                                                                  dense_Dense2[0][0]               
__________________________________________________________________________________________________
 dense_Dense3 (Dense)            [null,2,10]          210         concatenate_Concatenate1[0][0]   
==================================================================================================
Total params: 310
Trainable params: 310
Non-trainable params: 0
__________________________________________________________________________________________________
</pre>

### Expand Dimensions
*expandDims* expands a dimension.

```js
const {serial, expandDims } = require('orb-tensorflowjs')

const input = tf.input({shape: [4, 4]})
const output = serial(
  expandDims(),
  tf.layers.dense({units: 10, activation: 'softmax'})
).apply(input)
const m = tf.model({inputs: input, outputs: output})
m.summary()
```
<pre>
_________________________________________________________________
 Layer (type)                 Output shape              Param #   
=================================================================
 input1 (InputLayer)          [null,4,4]                0         
_________________________________________________________________
 expand_dims_ExpandDims1 (Exp [null,1,4,4]              0         
_________________________________________________________________
 dense_Dense1 (Dense)         [null,1,4,10]             50        
=================================================================
Total params: 50
Trainable params: 50
Non-trainable params: 0
_________________________________________________________________
</pre>

### Combining Input and Layer APIs | Example 1
In the below example, the *split()* pseudo layers splits the input into a 2x2 grid. *flat()* Input API arranges transforms it into an array of grids. Each element is mapped to the dense layer using *mapTo()*. The output of dense is forwarded to the downstream layers.
```js
const {serial, parallel, split, mapTo, expandDims, input: orbinput } = require('orb-tensorflowjs')

const input = tf.input({shape: [4, 4]})
const dl = tf.layers.dense({units: 4})
const sl = split({factor: 2, axis: 2})
const output = serial(
  split({factor: 2, axis: 1}),
  mapTo(sl),
  orbinput.flat(),
  mapTo(dl),
  tf.layers.concatenate(),
  tf.layers.flatten(),
  tf.layers.dense({units: 10, activation: 'softmax'})
).apply(input)
const m = tf.model({inputs: input, outputs: output})
m.summary()
```
<pre>
__________________________________________________________________________________________________
 Layer (type)                    Output shape         Param #     Receives inputs                  
==================================================================================================
 input1 (InputLayer)             [null,4,4]           0                                            
__________________________________________________________________________________________________
 split_Split2 (Split)            [[null,2,4],[null,2, 0           input1[0][0]                     
__________________________________________________________________________________________________
 split_Split1 (Split)            [[null,2,2],[null,2, 0           split_Split2[0][0]               
                                                                  split_Split2[0][1]               
__________________________________________________________________________________________________
 dense_Dense1 (Dense)            [null,2,4]           12          split_Split1[0][0]               
                                                                  split_Split1[0][1]               
                                                                  split_Split1[1][0]               
                                                                  split_Split1[1][1]               
__________________________________________________________________________________________________
 concatenate_Concatenate1 (Conca [null,2,16]          0           dense_Dense1[0][0]               
                                                                  dense_Dense1[1][0]               
                                                                  dense_Dense1[2][0]               
                                                                  dense_Dense1[3][0]               
__________________________________________________________________________________________________
 flatten_Flatten1 (Flatten)      [null,32]            0           concatenate_Concatenate1[0][0]   
__________________________________________________________________________________________________
 dense_Dense2 (Dense)            [null,10]            330         flatten_Flatten1[0][0]           
==================================================================================================
Total params: 342
Trainable params: 342
Non-trainable params: 0
__________________________________________________________________________________________________
</pre>

### Combining Input and Layer APIs | Example 2
Here, the *split()* input API splits the input array. *map()* pseudo layer applies 1:1 mapping of inputs to the arg layers.

```js
const {serial, map, input: orbinput } = require('orb-tensorflowjs')

const input1 = tf.input({shape: [4, 4]})
const input2 = tf.input({shape: [4, 4]})
const output = serial(
  orbinput.split(),
  map(
    tf.layers.dense({units: 10}),
    tf.layers.dense({units: 10}),
  ),
  tf.layers.concatenate(),
  tf.layers.flatten(),
  tf.layers.dense({units: 10, activation: 'softmax'})
).apply([input1, input2])
const m = tf.model({inputs: [input1, input2], outputs: output})
m.summary()
```
<pre>
__________________________________________________________________________________________________
 Layer (type)                    Output shape         Param #     Receives inputs                  
==================================================================================================
 input1 (InputLayer)             [null,4,4]           0                                            
__________________________________________________________________________________________________
 input2 (InputLayer)             [null,4,4]           0                                            
__________________________________________________________________________________________________
 dense_Dense1 (Dense)            [null,4,10]          50          input1[0][0]                     
__________________________________________________________________________________________________
 dense_Dense2 (Dense)            [null,4,10]          50          input2[0][0]                     
__________________________________________________________________________________________________
 concatenate_Concatenate1 (Conca [null,4,20]          0           dense_Dense1[0][0]               
                                                                  dense_Dense2[0][0]               
__________________________________________________________________________________________________
 flatten_Flatten1 (Flatten)      [null,80]            0           concatenate_Concatenate1[0][0]   
__________________________________________________________________________________________________
 dense_Dense3 (Dense)            [null,10]            810         flatten_Flatten1[0][0]           
==================================================================================================
Total params: 910
Trainable params: 910
Non-trainable params: 0
__________________________________________________________________________________________________
</pre>

### Combining Input and Layer APIs | Example 2
*repeat()* input API repeats the input and forwards the result. *map()* pseudo layer performs one-to-one mapping between input and layer args.
```js
const {serial, map, input: orbinput } = require('orb-tensorflowjs')

const input = tf.input({shape: [4, 4]})
const output = serial(
  orbinput.repeat(),
  map(
    tf.layers.dense({units: 10}),
    tf.layers.dense({units: 10}),
  ),
  tf.layers.concatenate(),
  tf.layers.dense({units: 10, activation: 'softmax'})
).apply(input)
const m = tf.model({inputs: input, outputs: output})
m.summary()
```
<pre>
__________________________________________________________________________________________________
 Layer (type)                    Output shape         Param #     Receives inputs                  
==================================================================================================
 input1 (InputLayer)             [null,4,4]           0                                            
__________________________________________________________________________________________________
 dense_Dense1 (Dense)            [null,4,10]          50          input1[0][0]                     
__________________________________________________________________________________________________
 dense_Dense2 (Dense)            [null,4,10]          50          input1[0][0]                     
__________________________________________________________________________________________________
 concatenate_Concatenate1 (Conca [null,4,20]          0           dense_Dense1[0][0]               
                                                                  dense_Dense2[0][0]               
__________________________________________________________________________________________________
 dense_Dense3 (Dense)            [null,4,10]          210         concatenate_Concatenate1[0][0]   
==================================================================================================
Total params: 310
Trainable params: 310
Non-trainable params: 0
__________________________________________________________________________________________________
</pre>
# Input APIs
These APIs transform the input arrangement. They apply to an input or an array of inputs. They **do not** modify tensors.

## flat
It flattens a nested input array. It also supports a depth parameter. The default *depth is 1*.
```js
const inputs = [t1, [t2, [t3]]] // t1, t2 and t3 are tensors.
const fi = flat()
const output = fi.apply(inputs)
```
