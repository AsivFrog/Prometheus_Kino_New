        // JavaScript function to toggle the selection of a number
        function toggleSelection(number) {
            var cell = document.getElementById("cell" + number);
            cell.classList.toggle("selected");

            // Get the form input for the clicked number
            var input = document.getElementById("selected_number" + number);

            // Check if the input exists
            if (input) {
                // Toggle the input value
                input.value = input.value === "" ? number : "";
            }

            // Check if there are any selected numbers
            var selectedNumbers = document.querySelectorAll('.selected');
            var submitButton = document.getElementById('submitButton');
            var counterElement = document.getElementById('counter');

            // Enable or disable the submit button based on the presence of selected numbers
            submitButton.disabled = selectedNumbers.length === 0;

            // Update the form with the currently selected numbers
            updateForm(selectedNumbers);

            // Display the updated counter value
            counterElement.innerText = selectedNumbers.length;
        }

        // Function to update the form with selected numbers
        function updateForm(selectedNumbers) {
            var form = document.getElementById('predictionForm');
            var inputContainer = document.getElementById('inputContainer');

            // Remove existing input elements
            inputContainer.innerHTML = '';

            // Dynamically generate inputs based on the clicked cells
            selectedNumbers.forEach(function (cell) {
                var number = cell.textContent;  // Get the number from the cell
                var input = document.createElement('input');
                input.type = 'hidden';
                input.name = 'selected_numbers[]';
                input.value = number;
                inputContainer.appendChild(input);
            });

            // Enable or disable the submit button based on the presence of selected numbers
            form.querySelector('button').disabled = selectedNumbers.length === 0;
