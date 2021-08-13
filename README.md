# orb-tensorflowjs
*orb-tensorflowjs* is a collection of APIs to facilitate concise model designs. They allow efficient interaction between inputs and the layers. [See some examples...](#model-design)

This library is organized into the following components:
* Input: Input component transforms an input array to another form. Some example APIs are: *split(), flat(), repeat()*. It **does not** operate on tensors. Input APIs are consumed, exclusively, by pseudo layers. [Jump to the docs...](#input-apis)
* Layer: Tensorflow layer. [Jump to the docs...](#layer-apis)
* Pseudo Layer: A Pseudo Layer defines the arrangement of inputs, layers and pseudo layers. The input, layer and pseudo layer APIs, all expose an *apply()* method which glues them together. [Jump to the docs...](#pseudo-layer-apis)
* Tensor: APIs to create and manipulate tensors. [Jump to the docs...](#tensor-apis)

These APIs work together to simplfy the model design. We will go through a few examples later on.

# Installation
Browser Installation. The module is exported as *orbtfjs* global variable.

```html
<script src="https://cdn.jsdelivr.net/npm/orb-tensorflowjs@latest/dist/index.js"></script>
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

<sub>
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
</sub>
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
<sub>
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
</sub>

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
<sub>
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
</sub>

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
<sub>
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
</sub>

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
<sub>
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
</sub>

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
<sub>
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
</sub>

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
<sub>
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
</sub>

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
<sub>
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
</sub>

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
<sub>
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
</sub>

# Input APIs
An input API operates on input arrays. They **do not** modify tensors. We explain them below with a few examples.

### flat
It flattens a nested input array. A depth parameter is supported to control the flattening behavior. The default *depth is 1*.
```js
// flat() without the depth.
const {input: orbinput } = require('orb-tensorflowjs')

const inputs = [t1, [t2, [t3]]] // t1, t2 and t3 are tensors.
const fi = orbinput.flat()
const output = fi.apply(inputs)
// Output: [t1, t2, [t3]]
```
```js
// flat() with the depth parameter.
const {input: orbinput } = require('orb-tensorflowjs')

const inputs = [t1, [t2, [t3]]] // t1, t2 and t3 are tensors.
const fi = orbinput.flat(depth = 2)
const output = fi.apply(inputs)
// Output: [t1, t2, t3]
```

### split
It splits an input. A factor parameter controls the number of splits. The default *factor is 1*.
```js
// Split into 2 pieces
const {input: orbinput } = require('orb-tensorflowjs')

const inputs = [t1, t2, t3] // t1, t2 and t3 are tensors.
const si = orbinput.split()
const output = si.apply(inputs)
// Output: [[t1, t2], [t3]]
```
```js
// Split into 3 pieces
const {input: orbinput } = require('orb-tensorflowjs')

const inputs = [t1, t2, t3] // t1, t2 and t3 are tensors.
const si = orbinput.split(3)
const output = si.apply(inputs)
// Output: [[t1], [t2], [t3]]
```

### repeat
It repeats an input. The result is placed in an array. A count parameter controls the number of repeasts. The default *count is 1*.
```js
// Repeat one more time
const {input: orbinput } = require('orb-tensorflowjs')

const input = t // A tensor
const ri = orbinput.repeat()
const output = ri.apply(input)
// Output: [t, t]
```
```js
// Repeat 3 more times
const {input: orbinput } = require('orb-tensorflowjs')

const input = t // A tensor
const ri = orbinput.repeat(3)
const output = ri.apply(input)
// Output: [t, t, t, t]
```

### func
*func()* allows custom manipulation of an input. It accepts a function argument.

```js
// Create three pieces of size: 2, 3, 2
const {input: orbinput } = require('orb-tensorflowjs')

const inputs = [t1, t2, t3, t4, t5, t6, t7] // A list of tensors
const fn = ts => [[t1, t2], [t3, t4, t5], [t6, t7]]
const fi = orbinput.func(fn)
const output = fi.apply(inputs)
// Output: [[t1, t2], [t3, t4, t5], [t6, t7]]
```

### tap
*tap()* is useful for diagnostic purposes during the model design process. It outputs the shape of incoming inputs. The incoming inputs are assumed to be a tensor or a list of tensors. For complex inputs or customized diagnostic information, it supports a *fn* parameter.
```js
// A single tensor example
const {input: orbinput} = require('orb-tensorflowjs')

const input = t // A tensor
const ti = orbinput.tap()
const output = ti.apply(input)
// Prints the shape of t
// Output: t
```
```js
// An array of tensors example
const {input: orbinput} = require('orb-tensorflowjs')

const inputs = [t1, t2] // t1 and t2 are tensors
const ti = orbinput.tap()
const output = ti.apply(inputs)
// Prints an array of shapes of tensors
// Output: [t1, t2]
```
# Layer APIs
These are a set of useful tensorflow layers. They operate on tensors.

### split
The *split()* layer splits a tensor into an array of tensors along an axis. An input with shape [4, 4, 4], when split along axis-0, results in 4 tensors. Each tensor has a shape: [1, 4, 4]. The *factor* and *axis* configurations control the split behavior. The default *axis is 1* and the default *factor is the size of the dimension represented by the axis*.

```js
const {split} = require('orb-tensorflowjs')

const input = tf.input({shape: [4, 2]})
const sl = split()
const output = sl.apply(input)
// Outputs an array of 4 tensors, each with shape [null, 1, 2]
// Output: [t1, t2, t3, t4]
```
```js
const {split} = require('orb-tensorflowjs')

const input = tf.input({shape: [4, 2]})
const sl = split({axis: 2})
const output = sl.apply(input)
// Outputs an array of 2 tensors, each with shape [null, 4, 1]
// Output: [t1, t2]
```

### expandDims
The *expandDims* layer expands input dimensions. An axis parameter allows to configure the axis of new dimension. The default *axis is 1*.

```js
const {expandDims} = require('orb-tensorflowjs')

const input = tf.input({shape: [4, 2]})
const ei = expandDims()
const output = ei.apply(input)
// Outputs a tensor with shape: [null, 1, 4, 2]
// Output: t
```

# Pseudo Layer APIs
The pseudo layers glue the model structure together. The operate on both, arrays and tensors. With the help of input and layer APIs, they enable concise and flexible model designs. We have posted [several examples here](#model-design)

### serial
*serial()* connects the argument layers sequentially. It can handle an arbitrary number of arg layers.

```js
const {serial} = require('orb-tensorflowjs')

const input = tf.input({shape: [4, 1]})
const sl = serial(
  tf.layers.reshape({targetShape: [2, 2]}),
  tf.layers.reshape({targetShape: [4, 1, 1]})
)
const output = sl.apply(input)
// Outputs a tensor with shape: [null, 4, 1, 1]
```
### parallel
*parallel()* connects the input with each arg layer. The output is an array containing the output of arg layers. It can handle an arbitrary number of arg layers.

```js
const {parallel} = require('orb-tensorflowjs')

const input = tf.input({shape: [4, 1]})
const pl = parallel(
  tf.layers.reshape({targetShape: [2, 2]}),
  tf.layers.reshape({targetShape: [4, 1, 1]})
)
const output = pl.apply(input)
// Outputs an array of tensors with shapes: [[null, 2, 2], [null, 4, 1, 1]]
```
### map
*map()* performs a **one-to-one** mapping between the inputs and the layers.

```js
const {map} = require('orb-tensorflowjs')

const inputs = [
  tf.input({shape: [4, 1]}),
  tf.input({shape: [1, 6]}),
]
const ml = map(
  tf.layers.reshape({targetShape: [2, 2]}),
  tf.layers.reshape({targetShape: [2, 3]}),
)
const output = ml.apply(inputs)
// Outputs an array of tensors with shapes: [[null, 2, 2], [null, 2, 3]]
```
### mapTo
*mapTo()* maps input elements to the same layer.

```js
const {mapTo} = require('orb-tensorflowjs')

const inputs = [
  tf.input({shape: [4, 1]}),
  tf.input({shape: [1, 4]}),
]
const ml = mapTo(tf.layers.reshape({targetShape: [2, 2]}))
const output = ml.apply(inputs)
// Outputs an array of tensors with shapes: [[null, 2, 2], [null, 2, 2]]
```

# Tensor APIs
A set of APIs to create and manipulate tensors.

### generate
*generate* supports several features.

**generate.lowerTriangular** generates a 2D lower triangular tensor. The size, the lower values and the upper values are configurable. The default *size is 2*, *lower value is 1* and the *upper value is 0*
```js
const {tensor} = require('orb-tensorflowjs')

const t = tensor.generate.lowerTriangular()
// Output shape: [2, 2]
//
// +---+---+
// | 1 | 0 |
// +---+---+
// | 1 | 1 |
// +---+---+
//
```
```js
const {tensor} = require('orb-tensorflowjs')

const t = tensor.generate.lowerTriangular(4, {lower: 2})
// Output shape: [2, 2]
//
// +---+---+---+---+
// | 2 | 0 | 0 | 0 |
// +---+---+---+---+
// | 2 | 2 | 0 | 0 |
// +---+---+---+---+
// | 2 | 2 | 2 | 0 |
// +---+---+---+---+
// | 2 | 2 | 2 | 2 |
// +---+---+---+---+
//
```
### random
*random* supports several APIs. They were primarily designed for **test purposes**. However, if need arises, they can be used in production.

**random.oneHot** generates one hot tensors. The size and the one hot dimension are configurable. The default values for *size is 1* and *dims is 10*. 
```js
// A tensor with default one hot dimensions (10)
const {tensor} = require('orb-tensorflowjs')

const size = 4
const t = tensor.random.oneHot(size)
// Outputs a one hot tensor with shape: [4, 10]
//
// +---+---+---+---+---+---+---+---+---+---+
// | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
// +---+---+---+---+---+---+---+---+---+---+
// | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
// +---+---+---+---+---+---+---+---+---+---+
// | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
// +---+---+---+---+---+---+---+---+---+---+
// | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 |
// +---+---+---+---+---+---+---+---+---+---+
//
```
```js
const {tensor} = require('orb-tensorflowjs')

const size = 4
const t = tensor.random.oneHot(size, {dims: 4})
// Outputs a one hot tensor with shape: [4, 4]
//
// +---+---+---+---+
// | 0 | 1 | 0 | 0 |
// +---+---+---+---+
// | 1 | 0 | 0 | 0 |
// +---+---+---+---+
// | 0 | 1 | 0 | 0 |
// +---+---+---+---+
// | 0 | 0 | 0 | 1 |
// +---+---+---+---+
//
```

**random.normalizedSample** creates a sample of values between 0 and 1. The size and the shape of the output is configurable. The default *sample size is 1* and the default *shape is [1]*. **This API is designed specifically for test purposes**.
```js
const {tensor} = require('orb-tensorflowjs')

const size = 5
const t = tensor.random.normalizedSample(size)
// Output shape: [5, 1]
```
```js
const {tensor} = require('orb-tensorflowjs')

const size = 5
const shape = [3, 4]
const t = tensor.random.normalizedSample(size, {shape: [3, 4]})
// Output shape: [5, 3, 4]
```
