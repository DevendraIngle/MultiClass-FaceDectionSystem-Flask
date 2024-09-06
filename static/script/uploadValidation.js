document.getElementById('fileElemTrigger').addEventListener('click', function() {
  document.getElementById('fileElem').click();
});

function handleFiles(files) {
  if (files.length > 0) {
      document.getElementById('file-uploaded-confirmation').classList.remove('hidden');
  }
}

function validateForm() {
  const name = document.getElementById('name').value;
  const fileInput = document.getElementById('fileElem');
  const errorMessage = document.getElementById('error-message');
  
  if (!name || fileInput.files.length === 0) {
      errorMessage.classList.remove('hidden');
      return false;
  }
  errorMessage.classList.add('hidden');
  return true;
}
