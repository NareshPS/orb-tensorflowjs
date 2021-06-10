const test = require('ava')
const {reduce, fill, range, ranges, resolveAddress} = require('orb-array')
const {constant} = require('orb-functions')
const {generate, random} = require('./tensor.js')

/////////////////////// random.oneHot [start] ///////////////////////
const ohSum = tni => tni.arraySync().map((oh => reduce.sum(oh)))

test('random.oneHot-no-args', t => {
  const tn = random.oneHot()

  t.deepEqual(tn.shape, [1, 10])
  t.deepEqual(ohSum(tn), [1])
})

test('random.oneHot-size', t => {
  const size = 5
  const tn = random.oneHot(size)

  t.deepEqual(tn.shape, [size, 10])
  t.deepEqual(ohSum(tn), fill(size, constant(1)))
})

test('random.oneHot-size-dims', t => {
  const [size, dims] = [5, 5]
  const tn = random.oneHot(size, {dims})

  t.deepEqual(tn.shape, [size, dims])
  t.deepEqual(ohSum(tn), fill(size, constant(1)))
})
/////////////////////// random.oneHot [end] ///////////////////////

/////////////////////// random.normalizedSample [start] ///////////////////////
const isNormalized = tn => {
  const values = tn.arraySync()
  const isnv = (v) => v>=0 && v<=1 // is the value normalized?
  const resolve = (address, container) => address.reduce((cc /**current container */, ai) => cc[ai], container)

  return ranges(...tn.shape).every((address) => isnv(resolve(address, values)))
}

test('random.normalizedSample-no-args', t => {
  const tn = random.normalizedSample()

  t.deepEqual(tn.shape, [1, 1])
  t.truthy(isNormalized(tn))
})

test('random.normalizedSample-size', t => {
  const size = 5
  const tn = random.normalizedSample(size)

  t.deepEqual(tn.shape, [5, 1])
  t.truthy(isNormalized(tn))
})

test('random.normalizedSample-size-shape', t => {
  const size = 5
  const shape = [3, 4]
  const tn = random.normalizedSample(size, {shape})

  t.deepEqual(tn.shape, [size, ...shape])
  t.truthy(isNormalized(tn))
})
/////////////////////// random.normalizedSample [end] ///////////////////////

/////////////////////// generate.lowerTriangular [start] ///////////////////////
const isTriangle = (tn, {lower = 1, upper = 0} = {}) => {
  const value = tn.arraySync()
  const triangle = (address) => {
    const [row, col] = address
    const v = resolveAddress(value, address)

    return col <= row ? v == lower: v == upper
  }

  return ranges(...tn.shape).every((address) => triangle(address))
}

test('generate.lowerTriangular-no-args', t => {
  const tn = generate.lowerTriangular()

  t.deepEqual(tn.shape, [2, 2])
  t.truthy(isTriangle(tn))
})

test('generate.lowerTriangle-size', t => {
  const size = 5
  const tn = generate.lowerTriangular(size)

  t.deepEqual(tn.shape, [size, size])
  t.truthy(isTriangle(tn))
})

test('generate.lowerTriangle-size-lower', t => {
  const [size, lower] = [5, 2]
  const tn = generate.lowerTriangular(size, {lower})

  t.deepEqual(tn.shape, [size, size])
  t.truthy(isTriangle(tn, {lower}))
})

test('generate.lowerTriangle-size-lower-upper', t => {
  const [size, lower, upper] = [5, 2, 3]
  const tn = generate.lowerTriangular(size, {lower, upper})

  t.deepEqual(tn.shape, [size, size])
  t.truthy(isTriangle(tn, {lower, upper}))
})
/////////////////////// generate.lowerTriangular [end] /////////////////////////