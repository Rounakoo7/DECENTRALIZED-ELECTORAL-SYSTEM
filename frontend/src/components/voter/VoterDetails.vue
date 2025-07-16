<script setup>
const emit = defineEmits(["back", "continue"]);

const props = defineProps({
	voter: {
		type: Object,
		required: true,
	},
});

function formatDate(dob) {
	if (!dob) return "";
	const date = new Date(parseInt(dob));
	return date.toLocaleDateString("en-GB", {
		day: "2-digit",
		month: "long",
		year: "numeric",
	});
}
</script>

<template>
	<div
		class="w-full min-h-screen bg-gray-50 flex justify-center items-center p-4 md:p-10"
	>
		<div
			class="w-full max-w-6xl bg-white rounded-xl shadow-md overflow-hidden flex flex-col md:flex-row"
		>
			<!-- Image Section (40% width) -->
			<div class="w-full md:w-2/5 bg-gray-200">
				<img
					v-if="voter.imageUrl"
					:src="voter.imageUrl"
					alt="Voter"
					class="w-full h-full object-cover"
				/>
				<div
					v-else
					class="h-full flex items-center justify-center text-gray-400 text-lg font-medium"
				>
					No Image Available
				</div>
			</div>

			<!-- Details Section (60%) -->
			<div class="w-full md:w-3/5 p-10 space-y-6">
				<h2 class="text-4xl font-extrabold text-gray-900">{{ voter.name }}</h2>

				<div
					class="grid grid-cols-1 sm:grid-cols-2 gap-4 text-gray-800 text-lg"
				>
					<p><span class="font-semibold">Father Name:</span> {{ voter.fatherName }}</p>
          <p><span class="font-semibold">Gender:</span> {{ voter.sex }}</p>
					<p>
						<span class="font-semibold">DOB:</span> {{ formatDate(voter.dob) }}
					</p>
					<p><span class="font-semibold">UID:</span> {{ voter.uid }}</p>
					<p><span class="font-semibold">Phone:</span> {{ voter.phone }}</p>
					<p class="sm:col-span-2">
						<span class="font-semibold">Location:</span> {{ voter.location }}
					</p>
					<p class="sm:col-span-2">
						<span class="font-semibold">Constituency:</span>
						{{ voter.constituency }}
					</p>
				</div>

				<!-- Message Section -->
				<div
					class="bg-yellow-100 border border-yellow-300 text-yellow-800 p-4 rounded-lg text-base shadow-sm"
				>
					<p>
						<strong>Reminder:</strong>
						<span v-if="!voter.isVoted">
							If your details are valid, continue to proceed with voting.</span
						>
						<span v-else>
							You have already voted. Thank you for your participation.</span
						>
					</p>
				</div>

				<!-- Buttons -->
				<div class="flex gap-4 pt-2">
					<button
						@click="$emit('back')"
						class="bg-black hover:bg-gray-800 text-white px-6 py-3 rounded-lg text-lg font-semibold transition duration-200"
					>
						Back
					</button>
					<button
						v-if="!voter.isVoted"
						@click="$emit('continue')"
						class="bg-yellow-400 hover:bg-yellow-500 text-white px-6 py-3 rounded-lg text-lg font-semibold transition duration-200"
					>
						Continue
					</button>
				</div>
			</div>
		</div>
	</div>
</template>
