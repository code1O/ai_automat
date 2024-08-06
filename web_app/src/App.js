import './App.css';


import { HomeScreen } from './components/screens';

import { DraggableDialog } from './components/Dialogs';

function App() {
  return (
    <div>
      <DraggableDialog 
      TextCancel='NO' TextOK='YES'
      OkEvent={console.log("true")} CancelEvent={console.log("false")}
      Title='Alert Draggable' Content='This is some testing content'
      />
      <HomeScreen/>
    </div>
  );
}

export default App;
