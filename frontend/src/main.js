import { createApp, ref } from 'vue';
import App from './App.vue';
import './assets/main.css';
import Toast from 'vue-toastification';
import 'vue-toastification/dist/index.css';
import router from './router';

const app = createApp(App);

app.use(Toast, {
    position: 'top-right',
    timeout: 5000,
    closeOnClick: true,
  });

app.use(router);
app.mount('#app');