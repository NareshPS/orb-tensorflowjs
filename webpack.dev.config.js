const path = require('path')

module.exports = {
  mode: 'development',
  entry: {
    index: "./src/index.js"
  },
  output: {
    filename: "[name].js",
    path: path.join(__dirname, 'dist'),
    library: "orbtfjs"
  },
  module: {
    rules: [
      {
        test: /\.js$/,
        enforce: 'pre',
        use: ["source-map-loader"],
      }
    ],
  },
  devtool : 'inline-source-map',
  resolve: {
    modules: ['node_modules']
  },
}
