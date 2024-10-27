"use client";

import React, { useState, useEffect, ComponentType, FormEvent } from "react";
import * as ScrollArea from "@radix-ui/react-scroll-area";

import styles from "../styles.module.css";

//                  @mui/material modules
// =================================================================
import Card from "@mui/material/Card";
import CardHeader from "@mui/material/CardHeader";
import CardContent from "@mui/material/CardContent";
import CardActions from "@mui/material/CardActions";
import Button from "@mui/material/Button";
import Input from "@mui/material/Input";
import TextField from "@mui/material/TextField"
// =================================================================

import { _Implement_Utils } from "../components/utils";

import Specifiers from "Globals/Specifiers.json";
import Miscs from "Globals/Miscs.json";

//                                          REACT ICONS
// =================================================================================================================
import { SlBubbles, SlActionUndo, SlHeart } from "react-icons/sl";
import { HiArrowSmRight, HiChartPie, HiInbox, HiShoppingBag, HiTable, HiUser, HiViewBoards } from "react-icons/hi";
import { FaArrowUp, FaArrowDown } from "react-icons/fa";
// =================================================================================================================

const DemoImages = Miscs.Images[0].Demo[0];
const Image_Post = DemoImages.PostComponent;

export const ImageResolutions = (ScreenType: string = "Large") => {
    const JsonInstance = Specifiers.Image_Resolutions[0];
    const { LargeScreen, ShortScreen } = { LargeScreen: JsonInstance.LargeScreen, ShortScreen: JsonInstance.ShortScreen };
    const { LargeDimensions, ShortDimensions } = { 
        LargeDimensions: [LargeScreen.Image_Width, LargeScreen.Image_Height],
        ShortDimensions: [ShortScreen.Image_Width, ShortScreen.Image_Height]
    }
    switch (ScreenType) {
        // [Width, Height] Array order in all cases
        case "Large":
            return LargeDimensions
        case "Short":
            return ShortDimensions
        default:
            console.log("Please enter a valid type");
    }
};

interface DetailsPost {
    Author: string,
    Image: string,
    Fotter: string,
    Dimensions_Image: any,
    Likes: number,
    Comments: number,
    Reposts: number
}

export function ComponentPost({Author, Image, Fotter, Dimensions_Image, Likes, Comments, Reposts}: DetailsPost) {

    const [ isClick, setClick ] = useState(false);
    
    const {
        A: [ QuantityLikes, setQuantityLikes ],
        B: [ QuantityReposts, setQuantityReposts ],
        C: [ QuantityComments, setQuantityComments ]
    } = { 
        A: useState(Likes), 
        B: useState(Reposts),
        C: useState(Comments)
    }

    const [ likeColor, setLikeColor ] = useState("");
    const [ isDefault, setIsDefault ] = useState(false);
    const [ isOpenActionView, setOpenActionView ] = useState(false);
    const [ isVisibleComponent, setVisibleComponent ] = useState(false);
    const [ showComponent, setShowComponent ] = useState<React.ReactNode | null>(null);

    // Add a calling API function
    const Like_ActionClick = () => {

        if ((isClick) && (isDefault)) {
            setQuantityLikes(QuantityLikes - 1);
            setLikeColor("")
        } else {
            setQuantityLikes(QuantityLikes + 1);
            setLikeColor("red")
        }

        if (Likes >= 10_000_000_000) alert("WOW, it shouldn't be that many likes!! ");
        setClick(!isClick);
        setIsDefault(!isDefault);
    }

    const CommentsPost = () => {
        const [ newComment, setNewComment ] = useState("");
        if (isOpenActionView) setShowComponent(null);
        if (!isOpenActionView) setTimeout(() => setOpenActionView(true), 5000);
        setOpenActionView(!isOpenActionView);
        let ArrayContent = Array.from({length: 50});

        // TO DELETE BEFORE UPLOADING TO PRODUCTION
        const testContent = ArrayContent.map(
            (_, i, a) => `Hello World number ${a.length - i} in div`
        );

        const handlePostComment = (e: React.FormEvent<HTMLTextAreaElement | HTMLInputElement>) => {
            setNewComment(e.currentTarget.value);
            testContent.push(newComment);
        }

        const handleFocus = (event: React.FocusEvent<HTMLInputElement>) => {
            event.preventDefault();
            event.target.scrollIntoView({behavior: "smooth", block: "nearest"});
        }

        return (
            <div className="flex flex-col ml-[-525px]">
                <ScrollArea.Root className={styles.ScrollAreaRoot}>
                    <ScrollArea.Viewport className={styles.ViewPort}>
                        <div style={{padding: "15px 20px"}}>
                            <div className={styles.Text}>Content</div>
                            {testContent.map((tag) => (
                                <div className={styles.Tag} key={tag}>
                                    {tag}
                                </div>
                            ))}
                        </div>
                    </ScrollArea.Viewport>
                    <ScrollArea.Scrollbar
                    className={styles.Scrollbar}
                    orientation="vertical"
                    >
                        <ScrollArea.Thumb className={styles.Thumb} />
                    </ScrollArea.Scrollbar>
                    <ScrollArea.Scrollbar
                    className={styles.Scrollbar}
                    orientation="horizontal"
                    >
                        <ScrollArea.Thumb className={styles.Thumb} />
                    </ScrollArea.Scrollbar>
                    <ScrollArea.Corner className={styles.Corner} />
                </ScrollArea.Root>
                <TextField 
                onInput={() => handlePostComment}
                autoComplete="off"
                id="standard-basic" 
                variant="outlined" 
                sx={{maxWidth: 400}}
                style={{backgroundColor: "rgb(25,25,25)"}}
                label="Comment"
                onFocus={handleFocus}
                InputProps={{ onFocus: (e) => e.preventDefault(), style: {color: "rgb(245, 245, 245)"} }}
                InputLabelProps={{style: {color: "rgb(245, 245, 245)"}}}
                inputMode="text"
                />
            </div>
        )
    }

    return (
        <div>
            <Card sx={{maxWidth: 470}} style={{backgroundColor: "rgb(25,25,25"}} >
                <CardHeader title={`${Author}'s post`} titleTypographyProps={{fontSize: "17px", color: "rgb(230, 230, 230)"}} />
                <CardContent className="flex flex-col gap-20">
                    <img src={Image} alt="Image" style={{width: Dimensions_Image[0], height: Dimensions_Image[1]}}/>
                    <p style={{color: "rgb(230, 230, 230"}}>{Fotter}</p>
                </CardContent>
                <CardActions className="flex justify-around items-center mr-[190px]" style={{maxHeight: "60px"}}>
                    <div style={{display: "flex", alignItems: "center"}}>
                        <Button size="large" style={{margin: 0}}><SlActionUndo/></Button>
                        <span className="ml-[5px] text-slate-400">{Reposts}</span>
                    </div>
                    <div style={{display: "flex", alignItems: "center"}}>
                        <Button size="large" style={{margin: 0, marginLeft: 2}} onClick={() => Like_ActionClick()}><SlHeart color={likeColor}/></Button>
                        <span className="ml-[5px] text-slate-400">{QuantityLikes}</span>
                    </div>
                    <div style={{display: "flex", alignItems: "center"}}>
                        <Button size="large" style={{margin: 0, marginLeft: 2}} onClick={() => setShowComponent(<CommentsPost/>)}><SlBubbles/></Button>
                        <span className="ml-[5px] text-slate-400">{Comments}</span>
                    </div>
                </CardActions>
            </Card>
            {showComponent}
        </div>
    )
}

export class HandleShowComponent {
    private Calling: _Implement_Utils["Calling"];
    private React_SetterFunction: Function
    constructor (CallingParams: _Implement_Utils["Calling"], React_Setter: React.Dispatch<React.SetStateAction<any>>) {
        this.Calling = CallingParams;
        this.React_SetterFunction = React_Setter;
    }
    public ByKeyboardEvent(key: string) {
        const handleEvent = (event: KeyboardEvent) => {
            if (event.ctrlKey && event.key === key) {
                event.preventDefault();
                this.React_SetterFunction((prev: any) => !prev);
            }
        }
        window.addEventListener(this.Calling.Function_Typifier, handleEvent);
        return (() => window.addEventListener(this.Calling.Function_Typifier, handleEvent))
    }
}

export function HotkeyDisplay (component: ComponentType) {
    const [ isVisibleComponent, setVisibleComponent ] = useState(false);
    const [ showComponent, setShowComponent ] = useState<React.ReactNode | null>(null);

    const setCallingParams = { Function_Typifier: "keydown" }
    const HandlerDisplay = new HandleShowComponent(setCallingParams, setVisibleComponent);
    useEffect(() => {
        HandlerDisplay.ByKeyboardEvent("s");
    }, []);
}

export default function demo() {

    return (
    <main>
        <div className="flex flex-col justify-center items-center mt-0 ml-[9%] gap-[50px]">
            <ComponentPost
                Author="Emily"
                Fotter=""
                Likes={34}
                Comments={20}
                Reposts={10}
                Image={Image_Post[0]}
                Dimensions_Image={ImageResolutions()}
                />
            <ComponentPost 
                Author="Lucia"
                Fotter="Just me n the beach"
                Likes={100}
                Comments={99}
                Reposts={40}
                Image={Image_Post[1]}
                Dimensions_Image={ImageResolutions()}
                />
        </div>
    </main>
    )
}