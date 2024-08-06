
/**
 * Prepare functions for **backend and frontend**
 * 
 * ## What includes inself?
 * - Interpreter of python code
 * - Recopile data from an element (like `input` or something else)
 * 
 * So, Happy coding, dev! ;)
*/


import useState from 'react';

/**
 * 
 * Access to an element and get its value
 * 
 * NOTE: The type of `num_of` must be int positive number
 * 
 * @param {*} classifier Id or class of the target element
 * @param {*} typeClassifier The type or tag name which is declared the classifier
 * @param {*} num_OF The number that identify the element
 * 
 */
export const GetElement_value = (classifier, typeClassifier, num_OF) => {
    const doc = document;
    const GetValue = (consider_type => {
        const GetById = doc.getElementById(classifier)[num_OF].value;
        const GetBy_Class = doc.getElementsByClassName(classifier)[num_OF].value;
        const GetBy_tagname = doc.querySelectorAll(classifier)[num_OF].value;
        switch (consider_type) {
            default:
                return GetBy_tagname;
            case 'id':
                return GetById;
            case 'class':
                return GetBy_Class;
        }
    });
    return GetValue(typeClassifier);
};

export const GetCurrentURL = () => {
    const [ currentURL, setCurrentURL ] = useState('');
    const NavigateURL = () => {
        setCurrentURL(window.location.href);
    } 
    const dictionary_url = {
        "PathURL": currentURL,
        "NavigateURL": NavigateURL
    };
    return dictionary_url;
};


/**
 * Interpreter of **python code**
 * 
 * ## How it works?
 * 
 * It access to the server `/run-python` which is actually a route
 * that works like a server tho.
 * 
 * then, read the data of `code` and execute the data
 * with `subprocess` python module
*/

// It costed me round of two neural brain burned. LOL
// I had to consult internet. Omega LOL

export async function RunPythonCode(code){
    const CurrentURL = GetCurrentURL();
    const Path_TO_PythonServer = `${CurrentURL.PathURL}/run-python`;
    try {
        const response = await fetch(Path_TO_PythonServer, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ code })
        });
        const data = await response.json();
        console.log(data);
    } catch (error) {
        console.error('ERROR: ', error);
    }
};