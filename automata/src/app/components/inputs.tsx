import { _HandleRegister, _HandleLogin } from "../Backend/ManageServer"

type inputProps =  {
    keyUpEvent?: () => void;
    onButtonClick?: () => void;
    textHolder: string;
}

export const SearchInput = ({keyUpEvent, onButtonClick, textHolder}: inputProps) => {
    return(
    <div className="w-full max-w-sm min-w-[480px] relative mt-16">
    <div className="relative w-full mt-16">
        <input
        className="w-full h-10 pr-11 pl-3 py-2 bg-transparent rounded-xl placeholder:text-slate-400 border border-slate-300 focus:border-slate-400 hover:border-slate-300 shadow-lg focus:shadow-md duration-200 ease transition focus:outline-none"
        placeholder={textHolder}
        onKeyUp={keyUpEvent}
        />
        <button
        className="absolute h-8 w-8 right-1 top-1 my-auto px-2 flex items-center bg-slate-800 rounded-lg hover:bg-slate-700"
        type="button"
        onClick={onButtonClick}
        >
        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="#FFF" className="w-8 h-8">
            <path stroke-linecap="round" stroke-linejoin="round" d="m21 21-5.197-5.197m0 0A7.5 7.5 0 1 0 5.196 5.196a7.5 7.5 0 0 0 10.607 10.607Z" />
        </svg>
        </button>
    </div>
    </div>
    )
}

export const ChatQueryInput = ({onButtonClick, textHolder}: inputProps) => {
    return (
        <div>
            <div>
                <input placeholder={textHolder} className=""/>
                <button onClick={onButtonClick} className=""></button>
            </div>
        </div>
    )
}

const SomeInput = (props: any) =>{
    return (
        <div className="relative w-full mt-4 ml-6">
            <input
            id={props.id}
            className="w-full rounded-none h-10 pl-3 py-2 pr-11 bg-transparent 
            border-b border-slate-500 placeholder:text-slate-100 duration-200  hover:border-slate-300
            text-slate-100 ease transition outline outline-width-2 outline-1 outline-transparent ml-2 focus:ring-0"
            placeholder={props.placeholder}
            type={props.typeInput}
            onChange={props.onChange}></input>
        </div>
    )
}

type actionAccount = {
    createAccount: () => void,
    keepAnonymous: () => void,
    onChanges: {
        EmailInput: Function,
        PhoneInput: Function,
        UserInput: Function,
        PasswordInput: Function
    },
}

export const AccountRegister = ({createAccount, keepAnonymous, onChanges}: actionAccount) => {
    return (
        <div className="max-w-[400px] relative mt-4">
            <SomeInput placeholder="Email..." type="email" onChange={onChanges.EmailInput}/>
            <SomeInput placeholder="Phone number..." type="tel" onChange={onChanges.PhoneInput}/>
            <SomeInput placeholder="Username..." type="text" onChange={onChanges.UserInput}/>
            <SomeInput placeholder="Password..." type="password" onChange={onChanges.PasswordInput}/>
            <div className="flex flex-row justify-evenly ml-[-10px] mt-6">

                <button
                className="right-1 top-1 my-auto px-2 flex 
                bg-slate-800 rounded-lg hover:bg-slate-700"
                type="button"
                onClick={createAccount}
                >Create account</button>

                <button
                className="right-1 top-1 my-auto px-2 flex 
                bg-slate-800 rounded-lg hover:bg-slate-700"
                type="button"
                onClick={keepAnonymous}
                >Continue as anonymous</button>
                
            </div>
        </div>
    )
}

type loginAccount = {
    ForgotData: () => void,
    LoginIntoApp: () => void,
    onChanges: {
        UsernameInput: Function,
        PasswordInput: Function,
    }
}

export const AccountLogin = ({LoginIntoApp, ForgotData, onChanges}: loginAccount) => {
    return (
        <div className="max-w-[400px] relative mt-4">
            <SomeInput placeholder="Username..." typeInput="text" onChange={onChanges.UsernameInput} />
            <SomeInput placeholder="Password..." typeInput="password" onChange={onChanges.PasswordInput} />
            <div className="flex flex-row justify-evenly ml-[-10px] mt-6">
                <button
                className="right-1 top-1 my-auto px-2 flex bg-slate-800 rounded-lg hover:bg-slate-700"
                type="submit"
                onClick={LoginIntoApp}
                >Login Account</button>
                <button
                className="right-1 top-1 my-auto px-2 flex bg-slate-800 rounded-lg hover:bg-slate-700"
                type="button"
                onClick={ForgotData}
                >Forgot my data</button>
            </div>
        </div>
    )
}