require("@nomicfoundation/hardhat-toolbox");
require("dotenv").config();
// require("@nomiclabs/hardhat-waffle");
// require("@nomiclabs/hardhat-ethers");
const { ALCHEMY_API_URL, PRIVATE_KEY } = process.env;

module.exports = {
  defaultNetwork: "sepolia",
  networks: {
    hardhat: {
      chainId: 1337
    },
    sepolia: {
      url: ALCHEMY_API_URL,
      accounts: [`0x${PRIVATE_KEY}`],
    },
  },
  solidity: {
    version: "0.8.28",
    settings: {
      optimizer: {
        enabled: true,
        runs: 200
      }
    }
  }
};
