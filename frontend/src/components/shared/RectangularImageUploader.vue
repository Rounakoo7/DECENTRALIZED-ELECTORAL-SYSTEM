<script setup>
import { ref } from 'vue';

const emit = defineEmits(['file-selected']);
const fileInput = ref(null);
const imagePreview = ref('');
const errorMessage = ref('');

function triggerFileInput() {
  fileInput.value?.click();
}

async function handleFileChange(event) {
  const file = event.target.files[0];
  if (!file) return;

  if (!file.type.startsWith('image/')) {
    errorMessage.value = 'Please select an image file';
    return;
  }

  if (file.size > 5 * 1024 * 1024) {
    errorMessage.value = 'Image size should be less than 5MB';
    return;
  }

  errorMessage.value = '';

  const reader = new FileReader();
  reader.onload = (e) => {
    imagePreview.value = e.target.result;
    emit('file-selected', file);
  };
  reader.readAsDataURL(file);
}

function removeImage() {
  imagePreview.value = '';
  emit('file-selected', '');
  if (fileInput.value) {
    fileInput.value.value = '';
  }
}

defineExpose({ removeImage });
</script>

<template>
  <div class="w-full">
    <div class="flex flex-col items-center">
      <!-- Rectangular Preview -->
      <div v-if="imagePreview" class="mb-3 relative">
        <img 
          :src="imagePreview" 
          alt="Preview" 
          class="w-48 h-32 object-cover rounded-lg border-2 border-gray-200"
        />
        <button 
          @click="removeImage" 
          type="button"
          class="absolute -top-2 -right-2 bg-red-500 text-white rounded-full p-1 hover:bg-red-600"
        >
          <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-x"><path d="M18 6 6 18"/><path d="m6 6 12 12"/></svg>
        </button>
      </div>

      <!-- Upload Placeholder -->
      <div v-if="!imagePreview" class="flex flex-col items-center justify-center w-48 h-32 border-2 border-dashed border-gray-300 rounded-lg bg-gray-50 hover:bg-gray-100 cursor-pointer" @click="triggerFileInput">
        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="text-gray-400 mb-1"><path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h7"/><line x1="16" x2="22" y1="5" y2="5"/><line x1="19" x2="19" y1="2" y2="8"/><circle cx="9" cy="9" r="2"/><path d="m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21"/></svg>
        <span class="text-xs text-gray-500">Upload Image</span>
      </div>

      <!-- Error Message -->
      <input 
        type="file" 
        ref="fileInput" 
        @change="handleFileChange" 
        accept="image/*" 
        class="hidden"
      />

      <div v-if="errorMessage" class="mt-2 text-sm text-red-500">
        {{ errorMessage }}
      </div>
    </div>
  </div>
</template>
