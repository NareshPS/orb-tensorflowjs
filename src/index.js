const input = require('./input.js')
const tensor = require('./tensor.js')

module.exports = {
  ...require('./layer.js'),
  input,
  tensor
}