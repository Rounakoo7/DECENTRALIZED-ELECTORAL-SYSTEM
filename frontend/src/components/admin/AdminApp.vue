<script setup>
import { provide, reactive, onMounted } from "vue";
import { useRouter } from "vue-router";
import { useToast } from "vue-toastification";
import { fetchConstituencies } from "@/services/admin";

const tabs = [
	{ name: "Voter Registration", path: "/voter-registration" },
	{ name: "Candidate Registration", path: "/candidate-registration" },
	{ name: "Constituency Registration", path: "/constituency-registration" },
	{ name: "Get Results", path: "/get-results" },
];

const router = useRouter();
function goBack() {
	router.replace("/");
}

const toast = useToast();

const appState = reactive({
	constituencies: [],
	loadConstituencies,
	sex: [
		{ id: "1", name: "Male" },
		{ id: "2", name: "Female" },
		{ id: "3", name: "Others" },
	],
	parties: [
		{ id: "1", name: "Team A" },
		{ id: "2", name: "Team B" },
		{ id: "3", name: "Team C" },
		{ id: "4", name: "Team D" },
		{ id: "5", name: "Team E" },
		{ id: "6", name: "Team F" },
	],
});

async function loadConstituencies() {
	try {
		const response = await fetchConstituencies();
		if (response.status === 200) {
			const data = await response.json();
			appState.constituencies = data.map((c, index) => ({
				id: (index + 1).toString(),
				name: `${c.stateCode}-${c.type}-${c.number}-${c.name}`,
			}));
		} else {
			toast.error(await response.text());
		}
	} catch (error) {
		console.error("Error loading constituencies:", error);
	}
};

onMounted(loadConstituencies);

provide("appState", appState);
</script>

<template>
	<div class="min-h-screen flex flex-col">
		<header class="bg-blue-400 shadow-md">
			<div class="max-w-full mx-auto">
				<div class="flex justify-between items-center py-4 px-10">
					<h1 class="text-2xl font-bold text-white">Admin Dashboard</h1>
					<button
						@click="goBack"
						class="hover:opacity-50 transition-opacity duration-300"
					>
						<img
							src="@/assets/img/back-icon.svg"
							alt="Button"
							class="w-10 h-10"
						/>
					</button>
				</div>
				<div class="flex border-b border-blue-300 bg-blue-500">
					<router-link
						v-for="tab in tabs"
						:key="tab.path"
						:to="tab.path"
						class="px-10 py-3 text-sm font-medium transition-colors duration-200"
						:class="[
							$route.path === tab.path
								? 'border-b-2 border-white text-white'
								: 'text-blue-300 hover:text-white ',
						]"
					>
						{{ tab.name }}
					</router-link>
				</div>
			</div>
		</header>

		<main class="flex-grow py-8">
			<div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
				<router-view />
			</div>
		</main>
	</div>
</template>
