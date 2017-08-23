function showDiv(evt, elem) {
    "use strict";
    var i;
    
    var elements = ["svr_options", "xgboost_options", "dropout_options"];
    for (i = 0; i < elements.length; i += 1) {
        document.getElementById(elements[i]).style.display = "none";
    }
    if (elem.value === "svr") {
        document.getElementById("svr_options").style.display = "block";
    } else if (elem.value === "xgboost") {
        document.getElementById("xgboost_options").style.display = "block";
    } else if (elem.value === "dropout") {
        document.getElementById("dropout_options").style.display = "block";
    }
}





function openPage(evt, pageName) {
    "use strict";
    // Declare all variables
    var i, tabcontent, tablinks;

    // Get all elements with class="tabcontent" and hide them
    tabcontent = document.getElementsByClassName("tabcontent");
    for (i = 0; i < tabcontent.length; i += 1) {
        tabcontent[i].style.display = "none";
    }

    // Get all elements with class="tablinks" and remove the class "active"
    tablinks = document.getElementsByClassName("tablinks");
    for (i = 0; i < tablinks.length; i += 1) {
        tablinks[i].className = tablinks[i].className.replace(" active", "");
    }

    // Show the current tab, and add an "active" class to the button that opened the tab
    document.getElementById(pageName).style.display = "block";
    evt.currentTarget.className += " active";
}


