<script setup>
import { onMounted, onBeforeUnmount } from 'vue'

onMounted(() => {
  window.history.pushState(null, '', window.location.href);

  const blockBack = () => {
    window.history.pushState(null, '', window.location.href);
  }

  window.addEventListener('popstate', blockBack);

  onBeforeUnmount(() => {
    window.removeEventListener('popstate', blockBack);
  });
});

</script>

<template>
    <div class="h-screen flex relative">
        <div class="hidden md:flex w-1/2">
			<img class="object-cover object-center w-full h-full"
             src="@/assets/img/home-banner.svg" alt="Logo" />
		</div>

        <div class="w-full md:w-1/2 flex items-center justify-center bg-white p-10">
            <div class="w-full max-w-md text-center">
                <h2 class="text-3xl font-bold mb-4">Decentralized E-Voting App</h2>
                <p class="text-gray-600 text-base">
                Experience secure, transparent, and tamper-proof voting powered by blockchain technology.<br />
                Empowering democracy with trustless digital solutions — vote anytime, from anywhere.
                </p>

                <div class="space-y-6 mt-10">
                    <router-link 
                      to="/admin" 
                      class="w-full h-20 py-4 px-6 bg-blue-400 hover:bg-blue-500 text-white font-medium rounded-lg transition-colors flex items-center justify-center gap-2"
                      >
                      <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-shield"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10"/></svg>
                      Admin Portal
                    </router-link>
                    <router-link 
                      to="/voter" 
                      class="w-full h-20 py-4 px-6 bg-yellow-400 hover:bg-yellow-500 text-white font-medium rounded-lg transition-colors flex items-center justify-center gap-2"
                      >
                      <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-vote"><path d="m9 12 2 2 4-4"/><path d="M5 7c0-1.1.9-2 2-2h10a2 2 0 0 1 2 2v12H5V7Z"/><path d="M22 19H2"/></svg>
                      Voter Portal
                    </router-link>
                </div>

            </div>
        </div>
        
    </div>
  </template>