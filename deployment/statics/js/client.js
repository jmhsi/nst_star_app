var el = x => document.getElementById(x);


function showPickerContent(inputId) { el('content-input').click(); }

function showPickedContent(input) {
    el('content-upload-label').innerHTML = input.files[0].name;
    var reader = new FileReader();
    reader.onload = function (e) {
        el('content-image-picked').src = e.target.result;
        el('content-image-picked').className = '';
    }
    reader.readAsDataURL(input.files[0]);
}

function showPickerStyle(inputId) { el('style-input').click(); }

function showPickedStyle(input) {
    el('style-upload-label').innerHTML = input.files[0].name;
    var reader = new FileReader();
    reader.onload = function (e) {
        el('style-image-picked').src = e.target.result;
        el('style-image-picked').className = '';
    }
    reader.readAsDataURL(input.files[0]);
}


function stylize() {
    var contentFile = el('content-input').files;
    if (contentFile.length != 1) {
        alert('Please select 1 content image!');
    } 
    
    var styleFile = el('style-input').files;
    if (styleFile.length != 1) {
        alert('Please select 1 style image!');
    } 
    /*
    var ori_steps = el('steps').value;
    if (ori_steps < 1 || ori_steps > 150) {
        alert('Steps must be between 1 and 150!');
    }
    */
    
    el('stylize-button').innerHTML = 'Styling... Large images may take up to 1 min';
    
    var xhr = new XMLHttpRequest();
    var loc = window.location
   
    xhr.open('POST', `${loc.protocol}//${loc.hostname}:${loc.port}/stylize`, true);
    xhr.onerror = function() {alert (xhr.responseText);}
    
    var fileData = new FormData();
    fileData.append('content', contentFile[0]);
    fileData.append('style', styleFile[0]);
    /*
    figure out how to send steps without corrupting the style image?
    trying fileData.append('steps', steps) corrupts. Number(steps) corrupts
    */
    xhr.send(fileData);
    
    xhr.onload = function(e) {
        if (this.readyState === 4) {
            var response = JSON.parse(e.target.responseText);
            var byte_img = response['stylized_image']
            var height = response['h']
            var width = response['w']
            var steps = response['steps']
            el('stylized-image').style.width = width;
            el('stylized-image').style.height = height;
            el('stylized-image').className = '';
            el('stylized-image').src = "data:image/png;base64," + byte_img;
            el('stylize-button').innerHTML = 'Stylized. Rerun Style Transfer?';
            el('step-count-display').innerHTML = `Stylized for ${steps} steps`;
        }
    }
    
}

/*            
            showStylized()
function showStylized() {
    var reader = new FileReader()
    var file = el('stylized-image')
    reader.readAsDataURL(file.toDataURL)
}
*/