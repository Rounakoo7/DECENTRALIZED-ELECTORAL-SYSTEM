const BASE_URL = "http://localhost:5000/api/A";

export async function registerConstituency(constituency) {
    const response = await fetch(`${BASE_URL}/constituencies`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(constituency)
    });
    return response;
}

export async function registerCandidate(candidate) {
    const response = await fetch(`${BASE_URL}/candidates`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(candidate)
    });
    return response;
}

export async function registerVoter(voter) {
    const response = await fetch(`${BASE_URL}/voters`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(voter)
    });
    return response;
}

export async function fetchConstituencies() {
    const response = await fetch(`${BASE_URL}/constituencies`);
    if (response.status === 204) {
        console.log("No constituencies found.");
        return;
    }
    return response;
}

export async function fetchCandidates(constituency) {
    const response = await fetch(`${BASE_URL}/${constituency}/candidates`);
    if (response.status === 204) {
        console.log("No candidates found.");
        return;
    }
    return response;
}

export async function fetchVoters(constituency) {
	const response = await fetch(`${BASE_URL}/${constituency}/voters`);
	if (response.status === 204) {
		console.log("No candidates found.");
		return;
	}
	return response;
}

export async function fetchCandidate(id) {
    const response = await fetch(`${BASE_URL}/candidates/${id}`);
    if (response.status === 204) {
        console.log("Candidate not found.");
        return;
    }
    return response;
}

export async function fetchVoter(id) {
    const response = await fetch(`${BASE_URL}/voters/${id}`);
    if (response.status === 204) {
        console.log("Voter not found.");
        return;
    }
    return response;
}
