<script setup>
import { registerVoter } from "@/services/admin";
import { provide, ref, inject } from "vue";
import { useToast } from "vue-toastification";
import { uploadToPinata } from "@/services/pinata";
import ImageUploader from "../shared/ImageUploader.vue";
import RectangularImageUploader from "../shared/RectangularImageUploader.vue";

const appState = inject("appState");
const phonePattern = /^[6-9]\d{9}$/;
const loading = ref(false);
const toast = useToast();

const selectedFile = ref(null);
const profileImageUrl = ref('');
const imageUploaderRef = ref(null);

const irisSamples = ref([null, null, null]);
const irisSampleUrls = ref(['', '', '']);
const uploaderRefs = [ref(null), ref(null), ref(null)];

const voterForm = ref({
  name: "",
  fatherName: "",
  id: "",
  dob: "",
  phone: "",
  sex: "",
  constituency: "",
  address: "",
});

function handleFileSelected(file) {
  selectedFile.value = file;
};

function handleIrisFileSelected(index, file) {
  irisSamples.value[index] = file;
}

function validateForm() {
  return (
    voterForm.value.name &&
    voterForm.value.fatherName &&
    voterForm.value.id &&
    voterForm.value.dob &&
    phonePattern.test(voterForm.value.phone) &&
    voterForm.value.sex &&
    voterForm.value.constituency &&
    voterForm.value.address &&
    selectedFile.value &&
    irisSamples.value.every((f) => f)
  );
}

async function uploadAllIrisImages() {
  try {
    const urls = await Promise.all(
      irisSamples.value.map(async (file) => await uploadToPinata(file))
    );
    irisSampleUrls.value = urls;
  } catch (error) {
    throw new Error("Failed to upload iris images. " + error.message);
  }
}

async function uploadFileToPinata(file) {
  try {
    const url = await uploadToPinata(file.value);
    profileImageUrl.value = url;
  } catch (error) {
    console.log(error);
    throw new Error('Failed to upload image. Please try again. ->', error.message);
  }
}

async function register() {
  loading.value = true;
  try {
    await uploadFileToPinata(selectedFile);
    await uploadAllIrisImages();

    const voter = {
      name: voterForm.value.name,
      fatherName: voterForm.value.fatherName,
      imageUrl: profileImageUrl.value,
      sex: voterForm.value.sex,
      dob: dateToUnix(voterForm.value.dob),
      uid: voterForm.value.id,
      constituency: voterForm.value.constituency,
      location: voterForm.value.address,
      phone: voterForm.value.phone,
      irisSamples: irisSampleUrls.value
    };

    const response = await registerVoter(voter);
    const resultText = await response.text();

    if (response.status === 201) {
      toast.success(resultText);
      resetForm();
    } else {
      toast.error(resultText);
    }
  } catch (error) {
    console.error("Error registering voter:", error);
    toast.error(error.message);
  } finally {
    loading.value = false;
  }
}

function resetForm() {
  voterForm.value = {
    name: "",
    fatherName: "",
    id: "",
    dob: "",
    phone: "",
    sex: "",
    constituency: "",
    address: "",
  };
  selectedFile.value = null;
  profileImageUrl.value = '';
  irisSamples.value = [null, null, null];
  irisSampleUrls.value = ['', '', ''];
  imageUploaderRef.value?.removeImage();
  uploaderRefs.forEach((ref, index) => {
    ref?.value?.removeImage();
    irisSamples.value[index] = null;
  });
}

function dateToUnix(dateStr) {
  return new Date(dateStr).getTime().toString();
}

provide("appState", appState);
</script>

<template>
  <div class="bg-white p-6 rounded-lg shadow-md">
    <div v-if="loading" class="fixed inset-0 flex items-center justify-center bg-white bg-opacity-70 z-50">
      <div class="w-12 h-12 border-4 border-blue-500 border-t-transparent rounded-full animate-spin"></div>
    </div>

    <h2 class="text-xl font-semibold mb-6 text-gray-800">Voter Registration</h2>

    <form @submit.prevent="register" class="space-y-6">
      <div class="grid grid-cols-1 md:grid-cols-2 gap-6">

        <div class="md:col-span-2 flex justify-center">
          <ImageUploader
            ref="imageUploaderRef"
            @file-selected="handleFileSelected"
          />
        </div>

        <div>
          <label class="block text-sm font-medium text-gray-700 mb-2">Voter Name</label>
          <input v-model="voterForm.name" placeholder="Enter Name" type="text" class="w-full px-4 py-2 border border-gray-300 rounded-md" required />
        </div>
        <div>
          <label class="block text-sm font-medium text-gray-700 mb-2">Father's Name</label>
          <input v-model="voterForm.fatherName" placeholder="Enter Father's Name" type="text" class="w-full px-4 py-2 border border-gray-300 rounded-md" required />
        </div>
        <div>
          <label class="block text-sm font-medium text-gray-700 mb-2">Voter UID</label>
          <input v-model="voterForm.id" placeholder="Enter Id" type="text" class="w-full px-4 py-2 border border-gray-300 rounded-md" required />
        </div>
        <div>
          <label class="block text-sm font-medium text-gray-700 mb-2">Date of Birth</label>
          <input v-model="voterForm.dob" placeholder="Enter DOB" type="date" class="w-full px-4 py-2 border border-gray-300 rounded-md" required />
        </div>
        <div>
          <label class="block text-sm font-medium text-gray-700 mb-2">Contact Number</label>
          <input v-model="voterForm.phone" placeholder="Enter Phone Number" type="tel" class="w-full px-4 py-2 border border-gray-300 rounded-md" required />
        </div>
        <div>
          <label class="block text-sm font-medium text-gray-700 mb-2">Sex</label>
          <select v-model="voterForm.sex" class="w-full px-4 py-2 border border-gray-300 rounded-md" required>
            <option disabled value="">Select Sex</option>
            <option v-for="sex in appState.sex" :key="sex.id" :value="sex.name">{{ sex.name }}</option>
          </select>
        </div>
        <div>
          <label class="block text-sm font-medium text-gray-700 mb-2">Constituency</label>
          <select v-model="voterForm.constituency" class="w-full px-4 py-2 border border-gray-300 rounded-md" required>
            <option disabled value="">Select Constituency</option>
            <option v-for="constituency in appState.constituencies" :key="constituency.id" :value="constituency.name">
              {{ constituency.name }}
            </option>
          </select>
        </div>
        <div class="md:col-span-2">
          <label class="block text-sm font-medium text-gray-700 mb-2">Address</label>
          <textarea v-model="voterForm.address" placeholder="Enter Full Address" rows="3" class="w-full px-4 py-2 border border-gray-300 rounded-md" required></textarea>
        </div>
      </div>

      <div class="mt-6">
        <label class="block text-sm font-medium text-gray-700 mb-6">Upload Iris Images</label>
        <div class="flex gap-2 justify-center">

          <RectangularImageUploader
            v-for="(_, index) in irisSamples"
            :key="index"
            :ref="el => uploaderRefs[index] = el"
            @file-selected="file => handleIrisFileSelected(index, file)"
          />

        </div>
      </div>

      <div class="flex justify-end">
        <button
          :disabled="!validateForm()"
          type="submit"
          class="px-6 py-2 bg-blue-500 hover:bg-blue-600 text-white font-medium rounded-md disabled:opacity-65 disabled:cursor-not-allowed mt-6"
        >
          Register Voter
        </button>
      </div>
    </form>
  </div>
</template>
