<script setup>
import { ref, inject } from "vue";
import { useToast } from "vue-toastification";
import { registerConstituency } from "@/services/admin";

const appState = inject("appState");

const states = [
	{ id: "1", name: "Andhra Pradesh", code: "AP" },
	{ id: "2", name: "Arunachal Pradesh", code: "AR" },
	{ id: "3", name: "Assam", code: "AS" },
	{ id: "4", name: "Bihar", code: "BR" },
	{ id: "5", name: "Chhattisgarh", code: "CG" },
	{ id: "6", name: "Goa", code: "GA" },
	{ id: "7", name: "Gujarat", code: "GJ" },
	{ id: "8", name: "Haryana", code: "HR" },
	{ id: "9", name: "Himachal Pradesh", code: "HP" },
	{ id: "10", name: "Jharkhand", code: "JH" },
	{ id: "11", name: "Karnataka", code: "KA" },
	{ id: "12", name: "Kerala", code: "KL" },
	{ id: "13", name: "Madhya Pradesh", code: "MP" },
	{ id: "14", name: "Maharashtra", code: "MH" },
	{ id: "15", name: "Manipur", code: "MN" },
	{ id: "16", name: "Meghalaya", code: "ML" },
	{ id: "17", name: "Mizoram", code: "MZ" },
	{ id: "18", name: "Nagaland", code: "NL" },
	{ id: "19", name: "Odisha", code: "OR" },
	{ id: "20", name: "Punjab", code: "PB" },
	{ id: "21", name: "Rajasthan", code: "RJ" },
	{ id: "22", name: "Sikkim", code: "SK" },
	{ id: "23", name: "Tamil Nadu", code: "TN" },
	{ id: "24", name: "Telangana", code: "TG" },
	{ id: "25", name: "Tripura", code: "TR" },
	{ id: "26", name: "Uttar Pradesh", code: "UP" },
	{ id: "27", name: "Uttarakhand", code: "UK" },
	{ id: "28", name: "West Bengal", code: "WB" },
];

const constituencyTypes = [
	{ id: "1", name: "Parliamentary Constituency", code: "PC" },
	{ id: "2", name: "Assembly Constituency", code: "AC" },
];

const loading = ref(false);
const toast = useToast();

const constituencyForm = ref({
	name: "",
	number: "",
	stateCode: "",
	constituencyType: "",
});

function validateForm() {
	return (
		constituencyForm.value.name &&
		constituencyForm.value.number &&
		constituencyForm.value.stateCode &&
		constituencyForm.value.constituencyType
	);
}

async function register() {
	loading.value = true;

	const constituency = {
		name: constituencyForm.value.name,
		number: constituencyForm.value.number,
		stateCode: constituencyForm.value.stateCode,
		type: constituencyForm.value.constituencyType,
	};

	try {
		var response = await registerConstituency(constituency);
		if (response.status === 201) {
			console.log("Constituency successfully registered!");
			toast.success(await response.text());
			appState.loadConstituencies();
		} else {
			toast.error(await response.text());
		}

		resetForm();
	} catch (error) {
		console.error("Error registering constituency:", error);
		toast.error(error.message);
	} finally {
		loading.value = false;
	}
}

function resetForm() {
	constituencyForm.value = {
		name: "",
		code: "",
		region: "",
		population: "",
		description: "",
	};
}
</script>

<template>
	<div class="bg-white p-6 rounded-lg shadow-md">
		<div
			v-if="loading"
			class="fixed inset-0 flex items-center justify-center bg-white bg-opacity-70 z-50"
		>
			<div
				class="w-12 h-12 border-4 border-blue-500 border-t-transparent rounded-full animate-spin"
			></div>
		</div>
		<h2 class="text-xl font-semibold mb-6 text-gray-800">
			Constituency Registration
		</h2>
		<form @submit.prevent="register" class="space-y-6">
			<div class="grid grid-cols-1 md:grid-cols-2 gap-6">
				<div>
					<label class="block text-sm font-medium text-gray-700 mb-1"
						>Constituency Name</label
					>
					<input
						v-model="constituencyForm.name"
						placeholder="Enter Constituency Name"
						type="text"
						class="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-emerald-500 focus:border-emerald-500"
						required
					/>
				</div>
				<div>
					<label class="block text-sm font-medium text-gray-700 mb-2"
						>Constituency Number</label
					>
					<input
						v-model="constituencyForm.number"
						placeholder="Enter Constituency Number"
						type="text"
						class="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-emerald-500 focus:border-emerald-500"
						required
					/>
				</div>
				<div>
					<label class="block text-sm font-medium text-gray-700 mb-2"
						>State</label
					>
					<select
						v-model="constituencyForm.stateCode"
						class="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-emerald-500 focus:border-emerald-500"
						required
					>
						<option disabled value="">Select State</option>
						<option v-for="state in states" :key="state.id" :value="state.code">
							{{ state.name }}
						</option>
					</select>
				</div>
				<div>
					<label class="block text-sm font-medium text-gray-700 mb-2"
						>Constituency Type</label
					>
					<select
						v-model="constituencyForm.constituencyType"
						class="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-emerald-500 focus:border-emerald-500"
						required
					>
						<option disabled value="">Select Constituency Type</option>
						<option
							v-for="ct in constituencyTypes"
							:key="ct.id"
							:value="ct.code"
						>
							{{ ct.name }}
						</option>
					</select>
				</div>
			</div>
			<div class="flex justify-end">
				<button
					@click.prevent="register()"
					:disabled="!validateForm()"
					type="submit"
					class="px-6 py-2 bg-blue-500 hover:bg-blue-600 text-white font-medium rounded-md disabled:opacity-65 disabled:cursor-not-allowed"
				>
					Register Constituency
				</button>
			</div>
		</form>
	</div>
</template>
