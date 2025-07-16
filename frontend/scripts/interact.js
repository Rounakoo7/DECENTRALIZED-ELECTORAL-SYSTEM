const hre = require("hardhat");

async function main() {
  const [deployer] = await hre.ethers.getSigners();

  const esAddress = "0x1bAe8D3F3f2A6918B39E5b7462E204aAc9066C30";
  const ES = await hre.ethers.getContractFactory("ES");
  const es = await ES.attach(esAddress);

  // Example: calling initializeParties (only admin can do this)
//   const tx = await es.initializeParties();
//   await tx.wait();
//   console.log("✅ Parties initialized");

  // Example: register a voter (must be admin)
//   const voterTx = await es.registerVoter(
//     "Alice", "Female", 20220622, "uid123", "WB-PC-71-Kolkata", '22345678', '87345678', "encrypted-phone"
//   );
//   await voterTx.wait();
//   console.log("✅ Voter registered");

  // const conTx = await es.registerConstituency(
  //   "WB", "PC", '71', "Kolkata"
  // );
//   await conTx.wait();
//   console.log("✅ Constituency registered");

  //Example: get voter details
  const candidate = await es.getCandidate("03");
  console.log("Candidate details:", candidate);

  // const candidate = await es.getCandidate("82");
  // console.log("📌 Candidate details:", candidate);

  const cons = await es.getConstituencies();
  console.log("📌 Constituencies:", cons);

}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
