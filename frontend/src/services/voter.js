const BASE_URL = "http://localhost:5000/api/V";

export async function fetchCandidates(constituency) {
	const response = await fetch(`${BASE_URL}/${constituency}/candidates`);
	if (response.status === 204) {
		console.log("No candidates found for this constituency.");
		return;
	}
	return response;
}

export async function fetchIrisSamples(constituency) {
    const response = await fetch(`${BASE_URL}/${constituency}/irisSamples`);
    if (response.status === 204) {
        console.log("No iris samples found for this constituency.");
        return;
    }
    return response;
}

export async function casteVote(voterUid, candidateUid) {
	const response = await fetch(`${BASE_URL}/vote`, {
		method: "POST",
		headers: { "Content-Type": "application/json" },
		body: JSON.stringify({
			voterUid: voterUid,
			candidateUid: candidateUid,
		}),
	});
	return response;
}

export async function matchIrisSamples(search_id, search_image, images) {
	const formData = new FormData();
	formData.append("search_id", search_id);
	formData.append("search_image", search_image);

	images.forEach((item, i) => {
		formData.append(`images[${i}][id]`, item.id);
		formData.append(`images[${i}][image]`, item.image);
	});

	const resp = await fetch("http://127.0.0.1:5050/match", {
		method: "POST",
		body: formData,
	});

	if (!resp.ok) {
		throw new Error(await resp.text());
	}
	return resp.json();
}
  