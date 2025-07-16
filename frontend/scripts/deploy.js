const hre = require("hardhat");

async function main() {
  
  const ESFactory = await hre.ethers.getContractFactory("ES");
  const es = await ESFactory.deploy();
  await es.waitForDeployment();

  const esAddress = await es.getAddress();
  console.log(`ES contract deployed at: ${esAddress}`);

}

main().catch((error) => {
  console.error("Deployment error:", error);
  process.exitCode = 1;
});

