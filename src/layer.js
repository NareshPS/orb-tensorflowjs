const tf = require('./tfinit.js')
const { range } = require('orb-array')

const serial = (...ls) => ({
  apply: x => ls.reduce((rx, l) => l.apply(rx), x)
})

const parallel = (...ls) => ({
  apply: x => ls.length > 0? ls.map((l) => l.apply(x)): [x]
})

const mapTo = (l) => ({
  apply: inputs => l? inputs.map((input) => l.apply(input)): inputs
})

/**
 * It splits the input along the input dimension.
 * The result is an array with a size that equals split dimension
 */
class Split extends tf.layers.Layer {
  constructor({axis = 1, factor} = {}) {
    super({axis, factor})

    this.axis = axis
    this.factor = factor
  }

  computeOutputShape = is => {
    const os = [...is]
    const factor = this.factor || is[this.axis]
    os[this.axis] =  is[this.axis] / factor

    return range(factor).map(_ => os)
  }

  call = ([input]) => tf.split(input, this.factor || input.shape[this.axis], this.axis)

  getConfig() {
    return Object.assign({axis: this.axis, factor: this.factor}, super.getConfig())
  }

  static get className() {return "Split"}
}

/**
 * It expands input along the given axis.
 * 
 */
class ExpandDims extends tf.layers.Layer {
  constructor({axis = 1} = {}) {
    super({axis})
    
    this.axis = axis
  }

  computeOutputShape = is => {
    const prefix = is.slice(0, this.axis)
    const postfix = is.slice(this.axis)
    const os = [...prefix, 1, ...postfix]

    return os
  }

  call = ([input]) => input.expandDims(this.axis)
  getConfig() {
    return Object.assign({axis: this.axis}, super.getConfig())
  }
  static get className() {return "ExpandDims"}
}

const Layers = [ExpandDims, Split]

// Register custom layers to the serialization module to support model save/load APIs.
Layers.map((x) => tf.serialization.registerClass(x))

// Expose a function that creates the layer.
// It aligns the proramming style with tensorflowjs library.
const split = config => new Split(config)
const expandDims = config => new ExpandDims(config)

module.exports = {
  serial,
  parallel,
  split,
  mapTo,
  expandDims,
}