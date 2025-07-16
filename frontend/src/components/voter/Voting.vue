<script setup>
import { ref, onMounted, computed } from "vue";
import { useRoute, useRouter } from "vue-router";
import Candidates from "../shared/Candidates.vue";

const router = useRouter();
const route = useRoute();

const emit = defineEmits(["submitVote"]);

const constituency = ref(route.query.constituency || "WB-PC-71-Burdwan");
const remainingTime = ref(300);

const props = defineProps({
  candidates: {
    type: Array,
    required: true,
  }
});



onMounted(() => {
  const timer = setInterval(() => {
    if (remainingTime.value > 0) {
      remainingTime.value--;
    } else {
      clearInterval(timer);
      router.back();
    }
  }, 1000);
});

const formattedTime = computed(() => {
  const minutes = Math.floor(remainingTime.value / 60).toString().padStart(2, "0");
  const seconds = (remainingTime.value % 60).toString().padStart(2, "0");
  return `${minutes}:${seconds}`;
});

function handleVoteSubmit(selectedCandidateId) {
  emit("submitVote", selectedCandidateId);
}
</script>

<template>
  <div class="min-h-screen flex bg-gray-100">
    
    <div class="w-full md:w-1/4 bg-yellow-400 text-black p-6 flex flex-col justify-between items-center">
      <div>
        <h2 class="text-lg font-semibold">Constituency:</h2>
        <p class="text-xl font-bold mt-1">{{ constituency }}</p>
      </div>

      <div class="text-center mt-8">
        <p class="text-sm">Remaining Time</p>
        <p class="text-3xl font-bold tracking-wider">{{ formattedTime }}</p>
      </div>
    </div>

    <div class="flex-1 p-8">
      <Candidates :candidates="candidates" @submitVote="handleVoteSubmit" />
    </div>
  </div>
</template>
