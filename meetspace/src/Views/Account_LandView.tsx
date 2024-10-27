/**
 * 
 * HOME VIEW FOR ACCOUNT LOGIN/REGISTER
 * 
*/

import React, {useState, useEffect} from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeRaw from "rehype-raw";

import { CgLogIn } from "react-icons/cg"
import { DiGithubBadge } from "react-icons/di";
import { SiSpotify, SiGithub } from "react-icons/si";
import { HiBadgeCheck } from "react-icons/hi";
import { IoKeyOutline } from "react-icons/io5";
import { FaUserFriends } from "react-icons/fa";

import { __ManageAccount } from "../Backend/ManageUser";

//                  @mui/material modules
// =================================================================
import Card from "@mui/material/Card";
import CardHeader from "@mui/material/CardHeader";
import CardContent from "@mui/material/CardContent";
import CardActions from "@mui/material/CardActions";
import Button from "@mui/material/Button";
import Input from "@mui/material/Input";
// =================================================================

type _Handle_Account_Events = {
    Inputs_Register: {
        Email: Function,
        Username: Function,
        Password: Function
    },
    Inputs_Login: {
        Username: Function,
        Password: Function
    }
    Buttons_Login: {
        Login: () => void,
        Forgot_data: () => void
    }
}

type markdownTypes = {
    content: string
}

const MarkdownRenderer = ({content}: markdownTypes) => {
    return (
        <div className="markdown-body p-5 bg-gray-100 border border-gray-300 rounded-md max-w-3xl mx-auto font-sans">
            <ReactMarkdown 
            skipHtml={false}
            remarkPlugins={[remarkGfm]} 
            rehypePlugins={[rehypeRaw]}
            components={{
                h1: ({children}) => <h1 className="text-2xl font-bold mt-4">{children}</h1>,
                h2: ({children}) => <h2 className="text-xl font-bold mt-3">{children}</h2>,
                h3: ({children}) => <h3 className="text-lg font-bold mt-2">{children}</h3>,
                a: ({node, ...props}) => <a className="underline underline-offset-4 decoration-cyan-300 hover:decoration-transparent ease-out duration-200 visited:decoration-purple-600" target="_blank" {...props}/>,
                img: ({node, ...props}) => <img {...props}/>,
                li: ({node, ...props}) => <ul className="list-disc pl-5 space-y-2"><li {...props}/></ul>
            }}>
                {content}
            </ReactMarkdown>
        </div>
    )
};

export function RegisterView() {
    return (
        <div style={{display: "flex", justifyContent: "center", alignItems: "center", marginTop: "180px"}}>
            <Card sx={{ minWidth: 475, height: "320px" }}>
                <CardHeader title="Register your meetspace account" titleTypographyProps={{fontSize: "20px"}}/>
                <CardContent style={{ display: "flex", flexDirection: "column", gap: "20px" }}>
                    <Input type="Email" placeholder="Email..."/>
                    <Input type="Username" placeholder="Username..."/>
                    <Input type="Password" placeholder="Password..."/>
                </CardContent>
                <CardActions style={{justifyContent: "space-between", marginRight: "70px", marginTop: "20px"}}>
                    <Button size="small">Register</Button>
                    <Button size="small">Keep as anonymous</Button>
                </CardActions>
            </Card>
        </div>
    )
}

export function LoginView() {
    // const [username, setUsername] = useState("");
    return (
        <main>
            <div style={{display: "flex", justifyContent: "center", alignItems: "center", marginTop: "180px"}}>
                <Card sx={{ minWidth: 475, height: "250px" }}>
                    <CardHeader title="Login into your meetspace" titleTypographyProps={{fontSize: "20px"}} />
                    <CardContent style={{ display: "flex", flexDirection: "column", gap: "20px" }}>
                        <Input type="Text" placeholder="Username..." />
                        <Input type="Password" placeholder="Password..." />
                    </CardContent>
                    <CardActions style={{justifyContent: "space-between", marginRight: "100px"}}>
                        <Button size="large"><CgLogIn/>Login</Button>
                        <Button size="small">Register</Button>
                    </CardActions>
                </Card>
            </div>
        </main>
    )
}

const ContentMD = `
![React](https://img.shields.io/badge/react-%2320232a.svg?style=for-the-badge&logo=react&logoColor=%2361DAFB)

# Hello World as h1

This is a test content for markdown renderer

- list content
- [ ] TODO

[YouTube](https://youtube.com)

> [!NOTE]
> Hello World


`

export function _ProfileView() {
    return (
        <main>
            <MarkdownRenderer content={ContentMD}/>
        </main>
    )
}