<script setup>
import { ref, watch, computed } from "vue";
import { useToast } from "vue-toastification";

const props = defineProps({
  candidates: {
    type: Array,
    required: true,
  }
});

const emit = defineEmits(["submitVote"]);

const selectedCandidateId = ref(null);
const toast = useToast();

function handleSubmit() {
  if (!selectedCandidateId.value) {
    toast.warning("Please select a candidate to vote.");
    return;
  }
  
  emit("submitVote", selectedCandidateId);
}


</script>

<template>
  <div class="w-full max-w-5xl mx-auto px-4">
    <h2 class="text-3xl font-bold text-gray-800 mb-10 tracking-tight">Select Candidate To Cast Your Vote</h2>

    <div class="grid grid-cols-1 sm:grid-cols-2 gap-8">
      <div
        v-for="candidate in candidates"
        :key="candidate.uid"
        class="flex overflow-hidden rounded-2xl transition hover:scale-[1.01] border border-gray-200 hover:shadow-2xl"
        :class="{ 'ring-4 ring-yellow-400/60': selectedCandidateId === candidate.uid }"
      >
        <!-- Full-Height Left Image -->
        <div class="w-24 sm:w-32 h-full">
          <img
            :src="candidate.imageUrl"
            alt="Candidate"
            class="w-full h-full object-cover"
          />
        </div>

        <!-- Candidate Info -->
        <div class="flex-1 p-5 bg-white flex items-center justify-between">
          <div>
            <h3 class="text-lg font-semibold text-gray-900">{{ candidate.name }}</h3>
            <span
              class="inline-block bg-black text-white mt-1 text-xs px-3 py-1 rounded-full font-medium"
            >
              {{ candidate.politicalAffiliation }}
            </span>
          </div>

          <!-- Radio Button -->
          <input
            type="radio"
            :value="candidate.uid"
            v-model="selectedCandidateId"
            class="w-5 h-5 text-yellow-500 accent-yellow-400 cursor-pointer"
          />
        </div>
      </div>
    </div>

    <!-- Submit Button -->
    <div class="mt-10 text-center">
      <button
        @click="handleSubmit"
        :disabled="!selectedCandidateId"
        class="px-8 py-3 bg-yellow-400  hover:bg-yellow-500 text-white font-semibold rounded-full text-lg shadow-md transition disabled:opacity-50 disabled:cursor-not-allowed"
      >
        Submit Vote
      </button>
    </div>
  </div>
</template>

