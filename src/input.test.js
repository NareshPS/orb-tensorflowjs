const test = require('ava')
const { fill } = require('orb-array')
const {tap, flat, split, repeat, func} = require('./input.js')

/////////////////////////////// tap [start] ///////////////////////////////
test('tap-no-args', t => {
  const input = 'tap-input'
  const ti = tap()
  const output = ti.apply(input)

  t.is(output, input)
})

test('tap-with-fn', t => {
  const input = 'tap-input'
  const fn = _ => {}
  const ti = tap(fn)
  const output = ti.apply(input)

  t.is(output, input)
})
/////////////////////////////// tap [end] ///////////////////////////////

/////////////////////////////// flat [start] ///////////////////////////////
test('flat-no-args-single-input', t => {
  const input = 'flat-input'
  const fi = flat()
  const output = fi.apply(input)

  t.is(output, input)
})

test('flat-no-args-multiple-input', t => {
  const inputs = ['flat-input1', ['flat-input2', ['flat-input3']]]
  const fi = flat()
  const output = fi.apply(inputs)

  t.deepEqual(output, ['flat-input1', 'flat-input2', ['flat-input3']])
})

test('flat-with-depth', t => {
  const inputs = [
    'flat-input1',
    [ 'flat-input2', ['flat-input3', ['flat-input4']] ],
    [ 'flat-input5', ['flat-input6'] ]
  ]
  const fi = flat(2)
  const output = fi.apply(inputs)

  t.deepEqual(
    output,
    [ 'flat-input1', 'flat-input2', 'flat-input3', ['flat-input4'], 'flat-input5', 'flat-input6', ]
  )
})
/////////////////////////////// flat [end] ///////////////////////////////

/////////////////////////////// split [start] ///////////////////////////////
test('split-no-args-single input', t => {
  const input = 'split-input'
  const si = split()
  const output = si.apply(input)

  t.deepEqual(output, [[input], []])
})

test('split-no-args-multiple-inputs', t => {
  const inputs = ['split-input1', 'split-input2', 'split-input3']
  const si = split()
  const output = si.apply(inputs)

  t.deepEqual(output, [['split-input1', 'split-input2'], ['split-input3']])
})

test('split-with-factor', t => {
  const inputs = ['split-input1', 'split-input2', 'split-input3']
  const si = split(3)
  const output = si.apply(inputs)

  t.deepEqual(output, [['split-input1'], ['split-input2'], ['split-input3']])
})
/////////////////////////////// split [end] ///////////////////////////////

/////////////////////////////// repeat [start] ///////////////////////////////
test('repeat-no-args', t => {
  const input = 'input-repeat'
  const ri = repeat()
  const output = ri.apply(input)

  t.deepEqual(output, [input, input])
})

test('repeat-with-count', t => {
  const input = 'input-repeat'
  const ri = repeat(5)
  const output = ri.apply(input)

  t.deepEqual(output, fill(6, input))
})
/////////////////////////////// repeat [end] ///////////////////////////////

/////////////////////////////// func [start] ///////////////////////////////
test('func-no-args', t => {
  const input = 'input-func'
  const fi = func()
  const output = fi.apply(input)

  t.deepEqual(output, input)
})

test('func-with-fn', t => {
  const input = 'input-func'
  const fn = x => [x, x]
  const fi = func(fn)
  const output = fi.apply(input)

  t.deepEqual(output, fn(input))
})
/////////////////////////////// func [end] ///////////////////////////////