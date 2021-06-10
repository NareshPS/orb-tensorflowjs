const tf = require('./tfinit.js')
const { range, fill } = require('orb-array')
const { gobject } = require('orb-object')

const random = {
  oneHot: (size = 1, {dims = 10} = {}) => {
    const hiFunc = _ => Math.floor(Math.random() * dims) // random hot index function
    const his = range(size).map(hiFunc) // hot indices

    return tf.oneHot(his, dims)
  },

  normalizedSample: (size = 1, {shape = [1]} = {}) => tf.randomUniform([size, ...gobject(shape)], 0, 1)
}

const generate = {
  lowerTriangular: (size = 2, {lower = 1, upper = 0} = {})=> {
    const content = fill(
      size**2,
      (index) => {
        const rid = index / size // row id
        const cid = index % size // col id

        return cid <= rid? lower: upper
      }
    )

    return tf.tensor(content, [size, size])
  }
}

module.exports = {random, generate}