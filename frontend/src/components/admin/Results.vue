<script setup>
import { ref, inject } from 'vue';
import { fetchCandidates } from '@/services/admin';
import { useToast } from 'vue-toastification';

const toast = useToast();
const appState = inject("appState");

const selectedConstituency = ref('');
const candidates = ref([]);
const submitted = ref(false);
const loading = ref(false);

async function getResults() {
  try {
    loading.value = true;
    submitted.value = true;
    const response = await fetchCandidates(selectedConstituency.value);
    if (response.ok) {
      candidates.value = await response.json();
    } else {
      candidates.value = [];
      toast.error(await response.text());
    }
  } catch (error) {
    toast.error('Error fetching results');
    console.error(error);
  } finally {
    loading.value = false;
  }
}
</script>

<template>
  <div class="min-h-screen bg-gray-50 py-10 px-4 md:px-10">
    <div class="max-w-4xl mx-auto bg-white rounded-xl shadow-lg p-6 space-y-6">
      <h2 class="text-3xl font-bold text-center text-gray-800">View Election Result</h2>

      <div class="flex flex-col md:flex-row items-center gap-4 justify-center">
        <select
          v-model="selectedConstituency"
          class="w-full md:w-2/3 border rounded-md px-4 py-2 text-gray-700 shadow-sm"
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

        <button
          @click="getResults"
          :disabled="!selectedConstituency || loading"
          class="w-full md:w-auto bg-blue-500 hover:bg-blue-600 text-white font-medium px-6 py-2 rounded-md transition disabled:opacity-50"
        >
          <span v-if="loading">
            <svg
              class="animate-spin h-5 w-5 mr-2 inline-block text-white"
              xmlns="http://www.w3.org/2000/svg"
              fill="none"
              viewBox="0 0 24 24"
            >
              <circle
                class="opacity-25"
                cx="12"
                cy="12"
                r="10"
                stroke="currentColor"
                stroke-width="4"
              ></circle>
              <path
                class="opacity-75"
                fill="currentColor"
                d="M4 12a8 8 0 018-8v4l3-3-3-3v4a8 8 0 100 16v-4l-3 3 3 3v-4a8 8 0 01-8-8z"
              ></path>
            </svg>
            Loading...
          </span>
          <span v-else>See Result</span>
        </button>
      </div>

      <!-- Loader -->
      <div v-if="loading" class="text-center py-6 text-blue-600 font-medium">
        Fetching candidates...
      </div>

      <!-- Results -->
      <div v-if="!loading && candidates.length" class="grid grid-cols-1 md:grid-cols-2 gap-6 pt-4">
        <div
          v-for="candidate in candidates"
          :key="candidate.uid"
          class="p-4 border rounded-lg shadow-md flex items-center gap-4"
        >
          <img
            :src="candidate.imageUrl"
            alt="Candidate"
            class="w-20 h-20 object-cover rounded-lg"
          />
          <div>
            <h3 class="text-lg font-semibold text-gray-800">{{ candidate.name }}</h3>
            <p class="text-sm text-gray-500">{{ candidate.politicalAffiliation }}</p>
            <p class="text-md font-semibold text-blue-600">Votes: {{ candidate.votes }}</p>
          </div>
        </div>
      </div>

      <p
        v-else-if="submitted && !loading && !candidates.length"
        class="text-center text-gray-500 text-lg mt-6"
      >
        No candidates found for this constituency.
      </p>
    </div>
  </div>
</template>

<style scoped>
select:invalid {
  color: gray;
}
</style>
