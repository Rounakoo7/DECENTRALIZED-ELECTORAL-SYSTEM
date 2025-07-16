<script setup>
import { ref, inject } from "vue";
import { useToast } from "vue-toastification";
import { registerCandidate } from "@/services/admin";
import { uploadToPinata } from "@/services/pinata";
import ImageUploader from "../shared/ImageUploader.vue";

const appState = inject("appState");

const phonePattern = /^[6-9]\d{9}$/;

const loading = ref(false);
const toast = useToast();

const selectedFile = ref(null);
const ipfsUrl = ref('');

const imageUploaderRef = ref(null)

const candidateForm = ref({
	name: "",
	fatherName: "",
	id: "",
	dob: "",
	party: "",
	constituency: "",
	phone: "",
	sex: "",
	address: ""
});

function handleFileSelected(file) {
  selectedFile.value = file;
}

function validateForm() {
	return (
		candidateForm.value.name &&
		candidateForm.value.fatherName &&
		candidateForm.value.id &&
		candidateForm.value.dob &&
		phonePattern.test(candidateForm.value.phone) &&
		candidateForm.value.sex &&
		candidateForm.value.constituency &&
		candidateForm.value.party &&
		candidateForm.value.address &&
		selectedFile.value
	);
}

async function register() {
	loading.value = true;

	try {

		await uploadFileToPinata(selectedFile);

		const candidate = {
			name: candidateForm.value.name,
			fatherName: candidateForm.value.fatherName,
			imageUrl: ipfsUrl.value,
			sex: candidateForm.value.sex,
			dob: dateToUnix(candidateForm.value.dob),
			uid: candidateForm.value.id,
			constituency: candidateForm.value.constituency,
			location: candidateForm.value.address,
			phone: candidateForm.value.phone,
			politicalAffiliation: candidateForm.value.party,
			votes: [],
		};

		var response = await registerCandidate(candidate);
		if (response.status === 201) {
			imageUploaderRef.value.removeImage();
			console.log("Candidate successfully registered!");
			toast.success(await response.text());
		} else {
			toast.error(await response.text());
		}

		resetForm();
	} catch (error) {
		console.error("Error registering candidate:", error);
		toast.error(error.message);
	} finally {
		loading.value = false;
	}
}

function resetForm() {
	candidateForm.value = {
		name: "",
		fatherName: "",
		id: "",
		dob: "",
		party: "",
		constituency: "",
		phone: "",
		sex: "",
		address: "",
	};
}

function dateToUnix(dateStr) {
  return new Date(dateStr).getTime().toString();
}

// function unixToDateInput(timestamp) {
//   const date = new Date(timestamp);
//   const yyyy = date.getFullYear();
//   const mm = String(date.getMonth() + 1).padStart(2, '0');
//   const dd = String(date.getDate()).padStart(2, '0');
//   return `${dd}-${mm}-${yyyy}`;
// }

async function uploadFileToPinata(file) {

  try {
    
    const url = await uploadToPinata(file.value);
    ipfsUrl.value = url

  } catch (error) {
    throw new Error('Failed to upload image. Please try again.');
  }
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
			Candidate Registration
		</h2>
		<form @submit.prevent="register" class="space-y-6">
			<div class="grid grid-cols-1 md:grid-cols-2 gap-6">
				<div class="md:col-span-2 flex justify-center">
					<ImageUploader
					ref="imageUploaderRef"
            			@file-selected="handleFileSelected"
          			/>
				</div>
				<div>
					<label class="block text-sm font-medium text-gray-700 mb-2"
						>Candidate Name</label
					>
					<input
						v-model="candidateForm.name"
						placeholder="Enter Name"
						type="text"
						class="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-emerald-500 focus:border-emerald-500"
						required
					/>
				</div>
				<div>
					<label class="block text-sm font-medium text-gray-700 mb-2"
						>Father's Name</label
					>
					<input
						v-model="candidateForm.fatherName"
						placeholder="Enter Father's Name"
						type="text"
						class="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-emerald-500 focus:border-emerald-500"
						required
					/>
				</div>
				<div>
					<label class="block text-sm font-medium text-gray-700 mb-2"
						>Candidate ID</label
					>
					<input
						v-model="candidateForm.id"
						placeholder="Enter Id"
						type="text"
						class="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-emerald-500 focus:border-emerald-500"
						required
					/>
				</div>
				<div>
					<label class="block text-sm font-medium text-gray-700 mb-2"
						>Date of Birth</label
					>
					<input
						v-model="candidateForm.dob"
						type="date"
						class="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-emerald-500 focus:border-emerald-500"
						required
					/>
				</div>
				<div>
					<label class="block text-sm font-medium text-gray-700 mb-2"
						>Party Affiliation</label
					>
					<select
						v-model="candidateForm.party"
						class="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-emerald-500 focus:border-emerald-500"
						required
					>
						<option disabled value="">Select Party</option>
						<option
							v-for="party in appState.parties"
							:key="party.id"
							:value="party.name"
						>
							{{ party.name }}
						</option>
					</select>
				</div>
				<div>
					<label class="block text-sm font-medium text-gray-700 mb-2"
						>Contact Number</label
					>
					<input
						v-model="candidateForm.phone"
						placeholder="Enter Phone Number"
						type="tel"
						class="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-emerald-500 focus:border-emerald-500"
						required
					/>
				</div>
				<div>
					<label class="block text-sm font-medium text-gray-700 mb-2"
						>Sex</label
					>
					<select
						v-model="candidateForm.sex"
						class="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-emerald-500 focus:border-emerald-500"
						required
					>
						<option disabled value="">Select Sex</option>
						<option v-for="sex in appState.sex" :key="sex.id" :value="sex.name">
							{{ sex.name }}
						</option>
					</select>
				</div>
				<div>
					<label class="block text-sm font-medium text-gray-700 mb-2"
						>Constituency</label
					>
					<select
						v-model="candidateForm.constituency"
						class="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-emerald-500 focus:border-emerald-500"
						required
					>
						<option disabled value="">Select Constituency</option>
						<option
							v-for="constituency in appState.constituencies"
							:key="constituency.id"
							:value="constituency.name"
						>
							{{ constituency.name }}
						</option>
					</select>
				</div>
				<div class="md:col-span-2">
					<label class="block text-sm font-medium text-gray-700 mb-2"
						>Address</label
					>
					<textarea
						v-model="candidateForm.address"
						rows="3"
						class="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-emerald-500 focus:border-emerald-500"
						required
					></textarea>
				</div>
			</div>
			<div class="flex justify-end">
				<button
					@click.prevent="register()"
					:disabled="!validateForm()"
					type="submit"
					class="px-6 py-2 bg-blue-500 hover:bg-blue-600 text-white font-medium rounded-md disabled:opacity-65 disabled:cursor-not-allowed"
				>
					Register Candidate
				</button>
			</div>
		</form>
	</div>
</template>
