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

### Feed the same input to multiple layers
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

### Expand Dimensions
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
# Input APIs
These APIs transform the input arrangement. They apply to an input or an array of inputs. They **do not** modify tensors.

## flat
It flattens a nested input array. It also supports a depth parameter. The default *depth is 1*.
```js
const inputs = [t1, [t2, [t3]]] // t1, t2 and t3 are tensors.
const fi = flat()
const output = fi.apply(inputs)
```
