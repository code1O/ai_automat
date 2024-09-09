"use client";

import Image from "next/image";
import Link from "next/link";
import { useState } from "react";
import { SearchInput } from "./components/inputs";
import { IconAlert } from "./components/alerts"
import { Mainview } from "./Views/Account_LandView";

const externalLinks = {
  Github: "https://github.com/code1O/ai_automat",
  TensorFlow: "https://github.com/tensorflow/tensorflow",
  OpenAI: "https://openai.com",
  Discord: "https://discord.com"
}

export default function Home() {
  const [componentShow, setComponentShow] = useState<React.ReactNode | null>(null)
  const handleAlert = () => {
    setComponentShow(<IconAlert AlertMessage="Message" timeout={3000}/>)
  }
  return (
    <main className="flex min-h-screen flex-col items-center justify-start p-24">

      <div className="flex flex-col items-center mb-10">
        <span className="bg-clip-text text-5xl text-transparent bg-gradient-to-tr from-cyan-400 to-violet-600" translate="no">
          Automata
        </span>
        <p className="text-xl bg-clip-text text-transparent bg-gradient-to-tr from-cyan-400 to-violet-600 mt-2">
          The house of thousands surprises
        </p>
      </div>
      
      <div className="flex flex-col items-center">
        <div className="flex items-start">
          <SearchInput textHolder="Search anything..." onButtonClick={handleAlert}/>
        </div>
        {componentShow}
      </div>
    </main>
  );
}