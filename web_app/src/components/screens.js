import '../App.css';
import Chatbot_logo from "../Assets/Chatbot_logo.png";

import { DraggableDialog } from './Dialogs';

import { 
    useNavigate,
    NavLink, Link,
    BrowserRouter as Router, Routes, Route
} from "react-router-dom";

function Navigate(path) {
    let instance = useNavigate();
    instance(path);
};

const RedirectNavigation = (props) => {
    return (
        <DraggableDialog 
        TextCancel='NO' TextOK='YES'
        OkEvent={ Navigate(props.path) } CancelEvent={ console.log('Dialog listened: False') }
        Title='Redirection to component' Content='Are you sure to visit that direction?'
        />
    );
};

const ChatBot = props => {
    return (
        <div className='Chatbot-Screen-Display'>
            <div className='Chatbot-Menu-Other_Models'></div>
            <div className='Chatbot-Menu-Navbar'></div>
            <div className='Chatbot-Content-Display'></div>
            <div className='Chatbot-Prompt-Options'></div>
        </div>
    );
};

const SystemRequests = () => {
    return (
        <div className='SystemRequests-Screen-Display'>
            <div className=''></div>
            <div className=''></div>
            <div className=''></div>
            <div className=''></div>
        </div>
    );
};

const WebAutomation = () => {
    return (
        <div className='WebAutomation-Screen-Display'>
            <div className=''></div>
            <div className=''></div>
            <div className=''></div>
            <div className=''></div>
        </div>
    );
};

const ContentClassification = () => {}

const SelectionMenuOptions = () => {

    return (
        <div className='Menu-Options'>
            <h2 id='Categorie-Text-Selection-Content'>Artificial Intelligence models</h2>
            <div className='Selection-Menu-Options'>
                <div id='Card-Selection-Content'>
                    <h3 id='Title-Selection-Content'>Chatbot, your personal Assistant </h3>
                    <Link to='/ChatBot'>
                        <img src={Chatbot_logo} alt='Chatbot image' id='Img-Selection-Content'/>
                    </Link>
                    <p>Made with TensorFlow, it'll help you with anything do you need</p>
                </div>
                <div id='Card-Selection-Content'>
                    <h3 id='Title-Selection-Content'>Content Classification</h3>
                    <Link to='/ContentClassification'>
                        <img src='' alt='Content Classification Image' id='Img-Selection-Content' />
                    </Link>
                    <p>Made with TensorFlow, it'll help you to recognize objects</p>
                </div>
            </div>
            <h2 id='Categorie-Text-Selection-Content' style={{color: ''}}>Tasks automation</h2>
            <div className='Selection-Menu-Options'>
                <div id='Card-Selection-Content'>
                    <h3 id='Title-Selection-Content'>Media requests</h3>
                    <Link to='/WebAutomation'>
                        <img src='' alt='Web automation image' id='Img-Selection-Content'/>
                    </Link>
                    <p>Able to help you to do some web tasks, like download YouTube videos or others</p>
                </div>
                <div id='Card-Selection-Content'>
                    <h3 id='Title-Selection-Content'>System requests</h3>
                    <Link to='/SystemRequests' >
                        <img src='' alt='' id='Img-Selection-Content' />
                    </Link>
                    <p>Incorporated System Commands (ISC)</p>
                </div>
            </div>
        </div>
    );
    
};

export const HomeScreen = (props) => {
    return (
        <Router>
            <SelectionMenuOptions />
            <Routes>
                <Route path='/' element={SelectionMenuOptions}/>
                <Route path='/Chatbot' element={ChatBot} />
                <Route path='/ContentClassification' element={ContentClassification} />
                <Route path='/WebAutomation' element={WebAutomation} />
                <Route path='/SystemRequests' element={SystemRequests} />
            </Routes>
        </Router>
    );
};
