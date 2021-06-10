const test = require('ava')
const { serial, parallel, mapTo, split, expandDims } = require('./layer.js')
const tf = require('./tfinit.js')

///////////////////////////// serial [start] ///////////////////////////// 
test('serial-no-args', t => {
  const sl = serial()
  const input = 'x'
  const output = sl.apply(input)
  
  t.is(output, input)
})

test('serial-single-layer', t => {
  const input = tf.input({shape: [4, 1]})
  const layers = [ tf.layers.reshape({targetShape: [2, 2]}) ]
  const sl = serial(...layers)
  const output = sl.apply(input)

  t.deepEqual(output.shape.slice(1), [2, 2])
})

test('serial-multiple-layers', t => {
  const input = tf.input({shape: [4, 1]})
  const layers = [
    tf.layers.reshape({targetShape: [2, 2]}),
    tf.layers.reshape({targetShape: [4, 1, 1]})
  ]
  const sl = serial(...layers)
  const output = sl.apply(input)

  t.deepEqual(output.shape.slice(1), [4, 1, 1])
})
///////////////////////////// serial [end] ///////////////////////////////

///////////////////////////// parallel [start] ///////////////////////////////
test('parallel-no-args', t => {
  const input = 'sabaidee'
  const pl = parallel()
  const [first, ...rest] = pl.apply(input)

  t.is(rest.length, 0)
  t.is(first, input)
})

test('parallel-single-layer', t => {
  const input = tf.input({shape: [4, 1]})
  const layers = [ tf.layers.reshape({targetShape: [2, 2]}) ]
  const pl = parallel(...layers)
  const [first, ...rest] = pl.apply(input)

  t.deepEqual(rest, [])
  t.deepEqual(first.shape.slice(1), [2, 2])
})

test('parallel-multiple-layers', t => {
  const input = tf.input({shape: [4, 1]})
  const layers = [
    tf.layers.reshape({targetShape: [2, 2]}),
    tf.layers.reshape({targetShape: [4, 1, 1]})
  ]
  const pl = parallel(...layers)
  const output = pl.apply(input)
  const [first, second, ...rest] = output

  t.is(output.length, layers.length)
  t.deepEqual(rest, [])
  t.deepEqual(first.shape.slice(1), [2, 2])
  t.deepEqual(second.shape.slice(1), [4, 1, 1])
})
///////////////////////////// parallel [end] ///////////////////////////////

///////////////////////////// mapTo [start] ///////////////////////////////
test('mapTo-no-args-empty-inputs', t => {
  const ml = mapTo()
  const output = ml.apply([])

  t.deepEqual(output, [])
})

test('mapTo-no-args-with-inputs', t => {
  const input = tf.input({shape: [4, 1]})
  const ml = mapTo()
  const output = ml.apply([input])
  const [first, ...rest] = output

  t.deepEqual(first.shape.slice(1), [4, 1])
  t.deepEqual(rest, [])
})

test('mapTo-single-input', t => {
  const input = tf.input({shape: [4, 1]})
  const ml = mapTo(tf.layers.reshape({targetShape: [2, 2]}))
  const output = ml.apply([input])
  const [first, ...rest] = output

  t.deepEqual(first.shape.slice(1), [2, 2])
  t.deepEqual(rest, [])
})

test('mapTo-multiple-inputs', t => {
  const inputs = [
    tf.input({shape: [4, 1]}),
    tf.input({shape: [1, 4]}),
  ]
  const ml = mapTo(tf.layers.reshape({targetShape: [2, 2]}))
  const output = ml.apply(inputs)
  const [first, second] = output

  t.deepEqual(first.shape.slice(1), [2, 2])
  t.deepEqual(second.shape.slice(1), [2, 2])
})
///////////////////////////// mapTo [end] ///////////////////////////////

///////////////////////////// split [start] ///////////////////////////////
test('split-no-args', t => {
  const input = tf.input({shape: [4, 2]})
  const sl = split()
  const output = sl.apply(input)

  t.is(output.length, 4)
  output.forEach(o => {
    t.deepEqual(o.shape.slice(1), [1, 2])
  })
})

test('split-with-factor', t => {
  const input = tf.input({shape: [4, 2]})
  const sl = split({factor: 2})
  const output = sl.apply(input)

  t.is(output.length, 2)
  output.forEach(o => {
    t.deepEqual(o.shape.slice(1), [2, 2])
  })
})

test('split-with-axis', t => {
  const input = tf.input({shape: [4, 2]})
  const sl = split({axis: 2})
  const output = sl.apply(input)

  t.is(output.length, 2)
  output.forEach(o => {
    t.deepEqual(o.shape.slice(1), [4, 1])
  })
})
///////////////////////////// split [end] ///////////////////////////////

///////////////////////////// expandDims [start] ///////////////////////////////
test('expandDims-no-args', t => {
  const input = tf.input({shape: [2, 2]})
  const edl = expandDims()
  const output = edl.apply(input)

  t.deepEqual(output.shape.slice(1), [1, 2, 2])
})

test('expandDims-with-axis', t => {
  const input = tf.input({shape: [2, 2]})
  const edl = expandDims({axis: 3})
  const output = edl.apply(input)

  t.deepEqual(output.shape.slice(1), [2, 2, 1])
})
///////////////////////////// expandDims [end] ///////////////////////////////