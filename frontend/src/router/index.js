import { createRouter, createWebHistory } from "vue-router";
import Home from "@/components/Home.vue";
import AdminApp from "@/components/admin/AdminApp.vue";
import VoterApp from "@/components/voter/VoterApp.vue";
import VoterRegistration from "@/components/admin/VoterRegistration.vue";
import CandidateRegistration from "@/components/admin/CandidateRegistration.vue";
import ConstituencyRegistration from "@/components/admin/ConstituencyRegistration.vue";
import Results from "@/components/admin/Results.vue";

const routes = [
	{
		path: "/",
		name: "Home",
		component: Home,
	},
	{
		path: "/voter",
		name: "VoterApp",
		component: VoterApp,
	},
	{
		path: "/admin",
		name: "AdminApp",
		component: AdminApp,
		children: [
			{
				path: "",
				redirect: { name: "VoterRegistration" },
			},
			{
				path: "/voter-registration",
				name: "VoterRegistration",
				component: VoterRegistration,
			},
			{
				path: "/candidate-registration",
				name: "CandidateRegistration",
				component: CandidateRegistration,
			},
			{
				path: "/constituency-registration",
				name: "ConstituencyRegistration",
				component: ConstituencyRegistration,
			},
			{
				path: "/get-results",
				name: "GetResults",
				component: Results,
			},
		],
	},
];
  
  const router = createRouter({
    history: createWebHistory(),
    routes
  });
  
  export default router;