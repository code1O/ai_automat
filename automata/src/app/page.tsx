"use client";

import Image from "next/image";
import Link from "next/link";
import { useState } from "react";
import { SearchInput } from "./components/inputs";
import { IconAlert } from "./components/alerts"
import { RegisterView, LoginView } from "./Views/Account_LandView";
import { useRouter } from "next/navigation";

function MainApp () {
  return (
    <main>
      {/** Display posts */}
      <div className="">
        <div className="">
          
        </div>
      </div>
    </main>
  )
}

function SetWebRoutes() {
  const router = useRouter();
  const renderComponent = () => {
  }
  return <>{renderComponent()}</>
}

export default function Home() {
  return (
    <RegisterView />
  );
}