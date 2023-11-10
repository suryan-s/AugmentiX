function submitForm(event) {
    event.preventDefault(); // Prevent the form from submitting in the traditional way

    // Collect form data
    let formData = {
        scriptPath: document.getElementById('datasetPath').value,
        outputPath: document.getElementById('outputPath').value,
        flipImage: document.getElementById('flipImage').value,
        blurImageKernel: document.getElementById('blurImage').value
    };

    // Convert form data to JSON
    let jsonData = JSON.stringify(formData);

    // Log form data in the console
    console.log('Form Data:', formData);

    // Call Eel function with the JSON data
    eel.starter(jsonData);
}