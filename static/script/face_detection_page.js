document.addEventListener("DOMContentLoaded", function () {
  const dropArea = document.getElementById("drop-area");
  const fileInput = document.getElementById("image");
  const uploadButton = document.getElementById("upload-button");
  const browse = document.getElementById("browse");
  const confirmationMessage = document.getElementById("confirmation-message");
  const openCameraButton = document.getElementById("open_camera");
  const cameraLogo = document.getElementById("camera-logo");

  const redirectToLiveFaceRecognition = () => {
    window.location.href = "/live_face_recognition";
  };

  openCameraButton.addEventListener("click", redirectToLiveFaceRecognition);
  cameraLogo.addEventListener("click", redirectToLiveFaceRecognition);

  browse.addEventListener("click", () => fileInput.click());

  ["dragenter", "dragover", "dragleave", "drop"].forEach((eventName) => {
    dropArea.addEventListener(eventName, preventDefaults, false);
  });

  function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
  }

  ["dragenter", "dragover"].forEach((eventName) => {
    dropArea.addEventListener(
      eventName,
      () => dropArea.classList.add("highlight"),
      false
    );
  });

  ["dragleave", "drop"].forEach((eventName) => {
    dropArea.addEventListener(
      eventName,
      () => dropArea.classList.remove("highlight"),
      false
    );
  });

  dropArea.addEventListener("drop", handleDrop, false);

  function handleDrop(e) {
    const dt = e.dataTransfer;
    const files = dt.files;
    handleFiles(files);
  }

  fileInput.addEventListener("change", (e) => {
    const files = e.target.files;
    handleFiles(files);
  });

  function handleFiles(files) {
    const file = files[0];
    if (file) {
      uploadButton.disabled = false;
      confirmationMessage.hidden = false;
    }
  }

  uploadButton.addEventListener("click", () => {
    const file = fileInput.files[0];
    if (file) {
      uploadFile(file);
    } else {
      alert("No file selected!");
    }
  });

  function uploadFile(file) {
    const url = "YOUR_UPLOAD_URL"; // Replace with your upload URL
    const formData = new FormData();
    formData.append("file", file);

    fetch(url, {
      method: "POST",
      body: formData,
    })
      .then((response) => response.json())
      .then((data) => {
        console.log(data);
        confirmationMessage.innerText = "Image uploaded successfully!";
        setTimeout(() => {
          confirmationMessage.hidden = true;
        }, 3000);
      })
      .catch((error) => {
        console.error(error);
      });
  }
});
