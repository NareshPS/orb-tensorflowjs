const tfload = _ => {
  try {
    return require('@tensorflow/tfjs-node-gpu')
  } catch {
    try {
      return require('@tensorflow/tfjs-node')
    } catch {
      try {
        return require('@tensorflow/tfjs')
      } catch {
        throw Error(
          `Couldn't find @tensorflow/tfjs-node-gpu, @tensorflow/tfjs-node or @tensorflow/tfjs. Please install tensorflow.`)
      }
    }
  }
}

module.exports = tfload()