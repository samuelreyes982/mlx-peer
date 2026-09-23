const steps = [
  ["Each device loads only its assigned layers.", "The exporter streams separate weight files for the Mac and iPhone without first allocating the entire model in memory. The Mac keeps the input embeddings and output head."],
  ["The Mac prepares the input.", "The Python coordinator converts your prompt into tokens and input embeddings. These numerical representations enter the first stage of the model on the iPhone."],
  ["The iPhone does real model computation.", "The Swift worker executes layers 0–11 in the recorded 27B partition. It keeps its weights and KV/recurrent state locally, then sends the resulting activations back over USB."],
  ["The Mac completes the pass.", "The Mac runs layers 12–63 and the output head, then selects the next token. The devices repeat the process as the answer grows, reusing their local model state."]
];
const buttons = [...document.querySelectorAll('.step')];
const explanation = document.getElementById('step-explanation');
const flow = document.querySelector('.flow-panel');
buttons.forEach((button, index) => {
  button.addEventListener('click', () => {
    buttons.forEach((item, i) => {
      item.classList.toggle('selected', i === index);
      item.setAttribute('aria-pressed', String(i === index));
    });
    flow.dataset.activeStep = String(index);
    explanation.querySelector('strong').textContent = steps[index][0];
    explanation.querySelector('p').textContent = steps[index][1];
  });
});
const copyButton = document.getElementById('copy-command');
copyButton.addEventListener('click', async () => {
  const command = 'git clone https://github.com/samuelreyes982/mlx-peer.git\ncd mlx-peer';
  const status = document.getElementById('copy-status');
  try {
    await navigator.clipboard.writeText(command);
    copyButton.textContent = 'Copied';
    status.textContent = 'Clone commands copied to your clipboard.';
    setTimeout(() => { copyButton.textContent = 'Copy'; }, 2200);
  } catch {
    copyButton.textContent = 'Select to copy';
    status.textContent = 'Copy is unavailable here. Select the commands in the code block to copy them.';
  }
});
