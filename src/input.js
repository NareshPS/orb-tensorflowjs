const array = require('orb-array')
const object = require('orb-object')
const functions = require('orb-functions')

const tap = (fn) => ({
  apply: x => {
    const fx = object.is.array(x)? x: [x]
    fn? fn(x): fx.forEach((fxi) => console.info(fxi.shape))

    return x
  }
})

const flat = (depth = 1) => ({
  apply: input => object.is.array(input)? input.flat(depth): input
})

const split = (factor = 2) => ({
  apply: input => array.split(object.is.array(input)? input: [input], factor)
})

const repeat = (count = 1) => ({
  apply: input => array.repeat(input, count)
})

const func = (fn = functions.self) => ({
  apply: input => fn(input)
})

module.exports = {tap, flat, func, split, repeat}