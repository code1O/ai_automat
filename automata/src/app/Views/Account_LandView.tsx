/**
 * 
 * HOME VIEW FOR ACCOUNT LOGIN/REGISTER
 * 
*/

import React, {useState, useEffect} from "react";
import { __ManageAccount } from "../Backend/ManageUser";
import { Button } from "@nextui-org/react";
import { AccountInputs } from "../components/inputs";
import { Card, CardHeader, CardBody, CardFooter, Divider, Image, Link, Input } from "@nextui-org/react";

export const Mainview = () => {
    return (
        <Card className="w-[500px] self-center items-center bg-clip-content bg-gradient-to-bl from-amber-400 to-red-500 rounded-lg">
            <CardHeader className="flex gap-3">
            <Image
            alt="nextui logo"
            height={40}
            radius="sm"
            src="https://avatars.githubusercontent.com/u/86160567?s=200&v=4"
            width={40}
            />
            <div className="flex flex-col">
                <p className="text-md"> Account management </p>
                <p className="text-small text-default-500">Automata.com</p>
            </div>
            </CardHeader>
            <CardBody>
                <p className="text-md">Register an account</p>
                <AccountInputs textHolder="Username"></AccountInputs>
            </CardBody>
            <Divider/>
            <CardFooter>
            <Link
            isExternal
            showAnchorIcon
            href="https://github.com/code1O/ai_automat"
            >
            Visit source code on GitHub.
            </Link>
            </CardFooter>
        </Card>
    )
}