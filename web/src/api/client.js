export async function sendLandmarks(landmarks) {
  const response = await fetch("http://localhost:8080/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ landmarks })
  });

  return response.json();
}
