<script setup>
import { ref, reactive, provide, onMounted } from 'vue';
import EnterUidForm from './EnterUidForm.vue';
import VoterDetails from './VoterDetails.vue';
import { useRouter } from 'vue-router';
import { fetchConstituencies } from "@/services/admin";
import { fetchCandidates } from '@/services/voter';
import { casteVote } from '@/services/voter';
import Voting from './Voting.vue';
import { useToast } from "vue-toastification";

const router = useRouter();
const voter = ref(null);
const candidates = ref(null);
const loading = ref(false);
const toast = useToast();

const appState = reactive({
	constituencies: [],
  loadConstituencies
});

function goBack() {
  router.back();
}

function handleVoterFetched(data) {
  voter.value = data;
}

function resetToForm() {
  voter.value = null;
}

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

async function fetchCandidateList() {
  try {

    loading.value = true
    const response = await fetchCandidates(voter.value.constituency);

    if (response.status === 200) {
			const data = await response.json();
      candidates.value = data;
		} else {
			toast.error(await response.text());
		}
  } catch (error) {
    console.error("Error while fetching candidates:", error);
  } finally{
    loading.value = false;
  }
}

async function vote(selectedCandidateId) {

  try{
    loading.value = true;
    const response = await casteVote(voter.value.uid, selectedCandidateId.value);
    const resultText = await response.text();

    if (response.status === 200) {
      toast.success(resultText);
    } else {
      toast.error(resultText);
    }

  } catch (error) {
    console.error("Error while casting vote:", error);
    toast.error(error.message);
  } finally {
    loading.value = false;
    router.back();
  }
}

function handleVoteSubmit(selectedCandidateId) {
 vote(selectedCandidateId)
}

onMounted(loadConstituencies);

provide("appState", appState);

</script>

<template>
  <div class="min-h-screen bg-gray-100 flex flex-col">

    <div v-if="loading" class="fixed inset-0 flex items-center justify-center bg-white bg-opacity-70 z-50">
      <div class="w-12 h-12 border-4 border-blue-500 border-t-transparent rounded-full animate-spin"></div>
    </div>

    <!-- Header -->
    <header class="bg-yellow-400 shadow-md">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div class="flex justify-between items-center py-4">
          <h1 class="text-2xl font-bold text-white">Voter Dashboard</h1>
          <button @click="goBack" class="hover:opacity-80 transition duration-300">
            <img src="@/assets/img/back-icon.svg" alt="Back" class="w-10 h-10" />
          </button>
        </div>
      </div>
    </header>

    <!-- Main Content -->
    <main class="flex-grow flex items-center justify-center px-4 md:px-8 py-10">
  <div class="w-full max-w-6xl bg-white rounded-3xl shadow-xl overflow-hidden">
    
    <Voting v-if="candidates && candidates.length > 0" 
      v-model:candidates="candidates" 
      @submitVote="handleVoteSubmit" />

    <VoterDetails 
      v-else-if="voter" 
      :voter="voter" 
      @back="resetToForm" 
      @continue="fetchCandidateList" 
    />

    <EnterUidForm 
      v-else 
      @voterFetched="handleVoterFetched" 
    />

  </div>
</main>

  </div>
</template>
