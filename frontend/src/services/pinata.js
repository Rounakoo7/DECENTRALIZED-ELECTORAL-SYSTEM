
const PINATA_API_KEY = process.env.VUE_APP_PINATA_API_KEY;
const PINATA_API_SECRET = process.env.VUE_APP_PINATA_API_SECRET;

export async function uploadToPinata(file) {
	try {
		const formData = new FormData();
		formData.append("file", file);

		const response = await fetch(
			"https://api.pinata.cloud/pinning/pinFileToIPFS",
			{
				method: "POST",
				body: formData,
				headers: {
					pinata_api_key: PINATA_API_KEY,
					pinata_secret_api_key: PINATA_API_SECRET,
				},
			}
		);

		if (!response.ok) {
			throw new Error(`Failed to upload to Pinata: ${response.statusText}`);
		}

		const result = await response.json();

		return `https://gateway.pinata.cloud/ipfs/${result.IpfsHash}`;
	} catch (error) {
		console.error("Error uploading to Pinata:", error);
		throw error;
	}
}
