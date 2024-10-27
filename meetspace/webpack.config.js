const path = require("path");

const TsconfigPathsPlugin = require("tsconfig-paths-webpack-plugin");

module.exports = {
    resolve: {
        plugins: [
            new TsconfigPathsPlugin({
                configFile: path.resolve(__dirname, "./tsconfig.json")
            })
        ],
        extensions: [".ts", ".tsx", ".js", ".json"],
        alias: {
            "@Globals": path.resolve(__dirname, "src/globals"),
            "@Scripts": path.resolve(__dirname, "src/Scripts")
        }
    },
    module: {
        rules: [
            {
                test: /\.module\.css$/,
                use: [
                    "style-loader",
                    {
                        loader: "css-loader",
                        options: {
                            modules: true
                        }
                    }
                ]
            }
        ]
    }
};