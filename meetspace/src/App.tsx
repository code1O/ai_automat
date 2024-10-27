import React, {useState} from "react";
import "./App.css"
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import { RegisterView, LoginView, _ProfileView } from "./Views/Account_LandView";

import demo from "./Views/demo_app";
import styles from "./styles.module.css";

function SetRoutes(){
  return (
    <Router>
      <Routes>
        <Route path="/Demo" Component={demo}/>
        <Route path="/Login" Component={LoginView}/>
        <Route path="/Register" Component={RegisterView}/>
        <Route path="/ProfileDemo" Component={_ProfileView}/>
      </Routes>
    </Router>
  )
}

function App() {
  return (
    <main>
      <SetRoutes />
    </main>
  )
}


export default App;