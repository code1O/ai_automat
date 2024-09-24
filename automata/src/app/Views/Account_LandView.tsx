/**
 * 
 * HOME VIEW FOR ACCOUNT LOGIN/REGISTER
 * 
*/

import React, {useState, useEffect} from "react";
import { AccountRegister, AccountLogin } from "../components/inputs";
import { Card, CardHeader, CardBody, Image, Link } from "@nextui-org/react";

import Styles from "../styles/style_infinite.module.css";

export const RegisterView = () => {
    const [ username, setUsername ] = useState('');
    const [ password, setPassword ] = useState('');
    const [ email, setEmail ] = useState('');
    const [ phone, setPhone ] = useState('');
    const handleRegister = async () => {
        try {
            const res = await fetch("/api/handler?action=register", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ Email: email, Phone: phone, USERNAME: username, USER_PASSWORD: password })
            });
            const data = await res.json();
            data.success? console.log("Registration successful") : console.log("Registration failed: " + data.error);
            
        } catch(Error) {
            console.error("An error occurred: ", Error);
            alert("Error occurred while registering! ");
        }
    }
    const InputChanges = {
        EmailInput: (e: any) => setEmail(e.target.value),
        PhoneInput: (e: any) => setPhone(e.target.value),
        UserInput: (e: any) => setUsername(e.target.value),
        PasswordInput: (e: any) => setPassword(e.target.value),
    }
    return (
        <main className="flex min-h-screen flex-col items-center justify-start p-24">
            <div className="">
                <Card className="w-[500px] self-center items-center bg-clip-content bg-gradient-to-tr from-slate-700 to-slate-900 rounded-lg mt-[120px] text-slate-200">
                <CardHeader className="flex gap-3">
                <Image
                alt="nextui logo"
                height={40}
                radius="md"
                src="https://avatars.githubusercontent.com/u/86160567?s=200&v=4"
                width={40}
                />
                <div className="flex flex-col">
                    <p className="text-md"> Register an account </p>
                    <p className="text-small text-default-500">LinkHub.com</p>
                </div>
                </CardHeader>
                <CardBody>
                    <AccountRegister
                    createAccount={() => handleRegister()}
                    keepAnonymous={()=> console.log("Keeping anonymous")}
                    onChanges={InputChanges}
                    ></AccountRegister>
                </CardBody>
                </Card>
            </div>
        </main>
    )
}

export const LoginView = () => {
    const [username, setUsername] = useState('');
    const [password, setPassword] = useState('');
    const handleLogin = async() => {
        try{
            const res = await fetch("/api/handler?action=login", {
                method: "POST",
                headers: { "Content-Type": "aplication/json" },
                body: JSON.stringify({ USERNAME: username, USER_PASSWORD: password })
            });

            const data = await res.json();
            data.success? console.log("Login successful!"): console.log("Login Failed!: " + data.error);

        } catch (Error) {
            console.error("Something was wrong on the login!", Error);
            alert("Something was wrong");
        }
    }

    const InputLoginChanges = {
        UsernameInput: (e: any) => setUsername(e.target.value),
        PasswordInput: (e: any) => setPassword(e.target.value)
    }

    return (
        <main className="flex min-h-screen flex-col items-center justify-start p-24">
            <div className="">
                <Card className="w-[500px] self-center items-center bg-clip-content bg-gradient-to-tr from-slate-700 to-slate-900 rounded-lg mt-[120px] text-slate-200">
                <CardHeader className="flex gap-3">
                <Image
                alt="nextui logo"
                height={40}
                radius="md"
                src="https://avatars.githubusercontent.com/u/86160567?s=200&v=4"
                width={40}
                />
                <div className="flex flex-col">
                    <p className="text-md"> Login in your account </p>
                    <p className="text-small text-default-500">MeetSpace.com</p>
                </div>
                </CardHeader>
                <CardBody>
                    <AccountLogin
                    LoginIntoApp={() => handleLogin()}
                    ForgotData={() => console.log("")}
                    onChanges={InputLoginChanges}
                    />
                </CardBody>
                </Card>
            </div>
        </main>
    )
}