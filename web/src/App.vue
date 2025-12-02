
<template>
  <HandGame />
</template>

<script>
import HandGame from "./components/HandGame.vue";

export default {
  components: { HandGame }
};
</script>

<style scoped>
:global(body) {
  background-color: rgb(255, 255, 255);
}


.logo {
  height: 6em;
  padding: 1.5em;
  will-change: filter;
  transition: filter 300ms;
}
.logo:hover {
  filter: drop-shadow(0 0 2em #646cffaa);
}
.logo.vue:hover {
  filter: drop-shadow(0 0 2em #42b883aa);
}
</style>
