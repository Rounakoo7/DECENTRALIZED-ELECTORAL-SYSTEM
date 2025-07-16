<script setup>
import { ref, inject } from "vue";
import { useRouter } from 'vue-router';
import { fetchVoter } from "@/services/admin";
import { useToast } from "vue-toastification";
import RectangularImageUploader from "../shared/RectangularImageUploader.vue";
import { fetchIrisSamples } from "@/services/voter";
import { matchIrisSamples } from "@/services/voter";

const router = useRouter();
const toast = useToast();
const appState = inject("appState");

const loading = ref(false);
const voterUid = ref(null);
const voter = ref(null);
const selectedConstituency = ref('');
const irisImage = ref(null);
const irisSamples = ref(null);
const flatImages = ref([]);

const emit = defineEmits(["voterFetched"]);

function handleIrisImageSelected(file) {
  irisImage.value = file;
}

function validateForm() {
  return (
    voterUid.value &&
    selectedConstituency.value &&
    irisImage.value
  );
}


async function continueToDetails() {
  loading.value = true;
  try {

    // 1. Fetch iris samples
    const irisResp = await fetchIrisSamples(selectedConstituency.value);
    if (!irisResp.ok) {
      toast.error(await irisResp.text());
      return;
    }
    const data = await irisResp.json();
    irisSamples.value = data;

    // 2. Build flatImages
    const tempList = [];
    for (const [id, urls] of Object.entries(data)) {
      for (const url of urls) {
        const blob = await fetch(url).then(r => r.blob());
        const file = new File([blob], `iris_${id}.jpg`, { type: blob.type });
        tempList.push({ id: Number(id), image: file });
      }
    }
    flatImages.value = tempList;

    // 3. Prepare match payload
    const images = flatImages.value;
    const search_image = irisImage.value;   // your File from uploader
    const search_id = voterUid.value;

    // 4. (Optional) Debug FormData before sending
    const formData = new FormData();
    formData.append("search_id", search_id);
    formData.append("search_image", search_image);
    images.forEach((item, i) => {
      formData.append(`images[${i}][id]`, item.id);
      formData.append(`images[${i}][image]`, item.image);
    });
    for (let [k, v] of formData.entries()) {
      console.log(k, v);
    }

    const matchResult = await matchIrisSamples(search_id, search_image, images);

    // 6. Handle match result
    if (matchResult.match) {
      toast.success("Iris match successful!");
      await fetchVoterDetails();
    } else {
      toast.error("Iris match failed or Voter not registered.");
    }

  } catch (err) {
    console.error(err);
    toast.error(err.message || "An error occurred.");
  } finally {
    loading.value = false;
  }
}

async function fetchVoterDetails() {
  loading.value = true;
  try {
    const response = await fetchVoter(voterUid.value);
    if (response.ok) {
      const data = await response.json();
      voter.value = data;
      emit("voterFetched", data);
    } else {
      toast.error(await response.text());
    }
  } catch (err) {
    toast.error("Error fetching voter data.");
    console.error(err);
  } finally {
    loading.value = false;
  }
}

</script>


<template>
  <div class="min-h-screen flex items-center justify-center bg-gray-50 px-4">
    <div class="bg-white rounded-xl shadow-md p-8 w-full max-w-md">
      <h2 class="text-2xl font-bold text-gray-800 mb-6">Enter Details</h2>

      <!-- UID Input -->
      <div class="mb-4">
        <label class="block text-sm font-medium text-gray-700 mb-2">Voter Id</label>
        <input
          type="text"
          v-model="voterUid"
          placeholder="Enter Voter Id"
          class="w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-yellow-400" required
        />
      </div>

      <div class="mb-4">
        <label class="block text-sm font-medium text-gray-700 mb-2">Constituency</label>
        <select
          v-model="selectedConstituency"
          class="w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-yellow-400" required
        >
          <option disabled value="">Select Constituency</option>
          <option
            v-for="c in appState.constituencies"
            :key="c.id"
            :value="c.name"
          >
            {{ c.name }}
          </option>
        </select>
      </div>

      <div class="mb-6">
        <RectangularImageUploader @file-selected="handleIrisImageSelected" />
      </div>

      <button
        :disabled="!validateForm()"
        @click="continueToDetails"
        class="w-full bg-yellow-400 hover:bg-yellow-500 text-black font-semibold py-2 px-4 rounded-lg transition disabled:opacity-65 disabled:cursor-not-allowed"
      >
        Continue
      </button>

      <!-- Loader -->
      <div v-if="loading" class="mt-6 text-center">
        <div class="loader border-4 border-t-4 border-yellow-400 rounded-full w-12 h-12 animate-spin mx-auto"></div>
        <p class="mt-2 text-gray-600">Loading voter details...</p>
      </div>
    </div>
  </div>
</template>

<style scoped>
.loader {
  border-color: #facc15 transparent transparent transparent;
}
</style>
