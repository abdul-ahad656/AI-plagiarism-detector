import React, { useState } from 'react';


export default function Signup(){
    const [form, setForm] = useState({name:"", email:"",password:""});
    const handleChange = (e) => {
    setForm({ ...form, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      await Signup(form);
      alert("Signup successful!");
    } catch (err) {
      alert("Signup failed!");
    }
  };
  return (
    <div>
        <h1>Signup</h1>
        <form action="" onSubmit={handleSubmit}>
            <label htmlFor="">Full Name</label>
            <input type="text" name="name" placeholder='Enter you Full Name' onChange={handleChange}/>
            <label htmlFor="">Email</label>
            <input type="email" name="email" placeholder='Enter your Email' onChange={handleChange} />
            <label htmlFor="">Password</label>
            <input type="password" name="password" placeholder='Enter your Password' onChange={handleChange}/>
            <button>Register</button>
        </form>
    </div>
  )
}